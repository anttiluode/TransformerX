"""Reproduce the interim GPU audit without loading checkpoints or retraining.

Run from any directory: python results/analyze_gpu_20261030.py
Requires numpy and matplotlib. Inputs are the committed runs/ artifacts.
The additive observations below are transcribed from Antti's live log, not a
completed test run. The report deliberately makes no cross-seed inference.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SILENT = ROOT / 'runs/silent-routing/loop-interaction/seed-7'
ADDITIVE = [(25,.3902,.805),(50,.2827,.890),(75,.2612,.900),
            (100,.3739,.837),(125,.2820,.885),(150,.2545,.905)]


def main(preview=None):
    report = json.loads((SILENT/'results.json').read_text())
    rows = list(csv.DictReader((SILENT/'trials.csv').open()))
    episodes = sorted({int(r['episode']) for r in rows})
    assert len(episodes) == 64
    assert len(rows) == 4*64*96
    for condition, metrics in report['test'].items():
        subset = [r for r in rows if r['feedback']==condition]
        assert len(subset)==64*96
        for source, column in [('normal','p_right'),('bayes','bayes_p_right'),
                               ('disconnected','disconnected_p_right'),('reset','reset_p_right')]:
            y = np.array([int(r['truth']) for r in subset])
            p = np.array([float(r[column]) for r in subset])
            assert np.isclose(((p>.5)==y).mean(),metrics[source]['all']['accuracy'])
            nll = -np.log(np.maximum(np.where(y,p,1-p),1e-12)).mean()
            # CSV stores only p_right; float32 softmax components need not sum
            # to exactly 1. Allow that rounding when reconstructing p_left.
            assert np.isclose(nll,metrics[source]['all']['nll'],atol=1e-6)
    all_rows = [r for r in rows if r['feedback']=='all']
    deltas = []
    ages = {key:[] for key in ('0','1','2','3','4','5-9','10+')}
    for episode in episodes:
        stream = sorted([r for r in all_rows if int(r['episode'])==episode], key=lambda r:int(r['trial']))
        def accuracy(column):
            return np.mean([(float(r[column])>.5)==int(r['truth']) for r in stream])
        deltas.append(accuracy('p_right')-accuracy('disconnected_p_right'))
        has_switched = False
        for r in stream:
            has_switched |= r['switched']=='True'
            if not has_switched:
                continue
            age = int(r['age'])
            group = str(age) if age<5 else '5-9' if age<10 else '10+'
            ages[group].append(r)
    # Paired episode bootstrap preserves dependence within each 96-trial stream.
    rng = np.random.default_rng(20261030)
    boot = np.asarray(deltas)[rng.integers(0,len(deltas),(10000,len(deltas)))].mean(axis=1)*100
    post_switch = {}
    for age, subset in ages.items():
        post_switch[age] = dict(count=len(subset), **{
            source:float(np.mean([(float(r[col])>.5)==int(r['truth']) for r in subset]))
            for source,col in [('normal','p_right'),('bayes','bayes_p_right'),
                               ('disconnected','disconnected_p_right')]})
    inputs = [SILENT/'results.json', SILENT/'trials.csv', SILENT/'validation.csv',
              ROOT/'runs/language-gpu/results.json', ROOT/'runs/vision-gpu/results.json']
    summary = dict(date='2026-10-30',training_seeds_completed=1,
        selected_step=report['selected_step'],episodes=len(episodes),trials_per_condition=len(all_rows),
        paired_read_benefit_pp=float(np.mean(deltas)*100),
        paired_episode_bootstrap_95_percentile_pp=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap_scope='Test episodes conditional on one trained checkpoint; not training-seed uncertainty',
        post_switch_excludes_initial_segment=post_switch,
        additive_log=dict(status='in progress; user-provided console observations only',
                          observations=[dict(step=s,val_nll=n,val_accuracy=a) for s,n,a in ADDITIVE]),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'results/GPU_INTERIM_2026-10-30.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    scores=report['test']['all']
    names=['No-history\nreference','State read\ndisconnected','Normal\nloop','Bayesian\nreference']
    sources=['marginal','disconnected','normal','bayes']
    values=[scores[k]['all']['accuracy']*100 for k in sources]
    bars=axes[0].bar(names,values,color=['#8b949e','#b27639','#176b8c','#518b57'],width=.65)
    axes[0].set(ylim=(0,103),ylabel='Clean test accuracy (%)',title='Completed interaction run · seed 7')
    axes[0].bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4)
    axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    val=list(csv.DictReader((SILENT/'validation.csv').open()))
    axes[1].plot([int(r['step']) for r in val],[float(r['val_accuracy'])*100 for r in val],
                 color='#176b8c',label='Interaction · completed',linewidth=2)
    axes[1].plot([r[0] for r in ADDITIVE],[r[2]*100 for r in ADDITIVE],
                 color='#b27639',marker='o',label='Additive · live log through step 150',linewidth=2)
    axes[1].axhline(80,color='#8b949e',linestyle='--',label='Population no-history reference (80%)')
    axes[1].set(ylim=(75,96),xlabel='Training step',ylabel='Validation accuracy (%)',
                title='Validation curves · not the test comparison')
    axes[1].legend(fontsize=9,loc='lower right');axes[1].grid(alpha=.15)
    fig.suptitle('TransformerX · interim GPU results · 30 October 2026',fontsize=15)
    fig.savefig(ROOT/'results/figures/gpu-interim-2026-10-30.svg',metadata={'Date':None})
    if preview is not None:
        fig.savefig(preview,dpi=120)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','additive_log')},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',type=Path,help='Optional PNG path for visual inspection')
    main(parser.parse_args().preview)
