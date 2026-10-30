"""Audit all nine completed GPU runs; rebuild aggregate tables and figure.
Run: python results/analyze_full_sweep_20261030.py
Requires NumPy and Matplotlib; does not load checkpoints or train models.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('interaction','additive','concat')
SEEDS=(7,17,27)
CONDITIONS=('all','informative_only','invariant_only','none')


def main(preview=None):
    reports={}; streams={}; hashes={}; flat=[]; signature=None
    for seed in SEEDS:
        for variant in VARIANTS:
            directory=ROOT/'runs/silent-routing'/f'loop-{variant}'/f'seed-{seed}'
            r=json.loads((directory/'results.json').read_text())
            assert r['completed_steps']==r['training']['steps']==1000
            assert r['training']['seed']==seed and r['model']['router']==variant
            sig=({k:v for k,v in r['model'].items() if k!='router'},
                 {k:v for k,v in r['training'].items() if k not in ('seed','out')})
            if signature is None: signature=sig
            assert sig==signature
            val=list(csv.DictReader((directory/'validation.csv').open()))
            best=min(val,key=lambda row:float(row['val_clean_nll']))
            assert r['selected_step']==int(best['step'])
            rows=list(csv.DictReader((directory/'trials.csv').open()))
            table={(x['feedback'],int(x['episode']),int(x['trial'])):x for x in rows}
            assert len(rows)==len(table)==4*64*96
            # Validate every saved source/group metric, not just headline accuracy.
            for condition in CONDITIONS:
                ordered=[table[condition,e,t] for e in range(64) for t in range(96)]
                y=np.array([int(x['truth']) for x in ordered])
                info=np.array([x['informative']=='True' for x in ordered])
                for source,col in [('normal','p_right'),('bayes','bayes_p_right'),
                                   ('disconnected','disconnected_p_right'),('reset','reset_p_right')]:
                    p=np.array([float(x[col]) for x in ordered])
                    for group,mask in [('all',np.ones(len(y),dtype=bool)),('informative',info),('invariant',~info)]:
                        score=r['test'][condition][source][group]
                        assert score['count']==int(mask.sum())
                        assert np.isclose(((p[mask]>.5)==y[mask]).mean(),score['accuracy'])
                        nll=-np.log(np.maximum(np.where(y[mask],p[mask],1-p[mask]),1e-12)).mean()
                        assert np.isclose(nll,score['nll'],atol=1e-6)
                assert r['test'][condition]['restoration_max_error']==0
            diagnostic_fields=('feedback','episode','trial','tone','rule','truth','outcome','informative','switched','age','returning')
            evidence=[tuple(table[key][f] for f in diagnostic_fields) for key in sorted(table)]
            if seed in streams: assert evidence==streams[seed]
            else: streams[seed]=evidence
            reports[seed,variant]=r
            row=dict(seed=seed,variant=variant,selected_step=r['selected_step'],
                     trainable_parameters=r['budget']['trainable_parameters'],seconds=r['seconds'])
            for condition in CONDITIONS:
                for source in ('normal','disconnected','reset','bayes','marginal'):
                    for metric in ('accuracy','nll','brier'):
                        row[f'{condition}_{source}_{metric}']=r['test'][condition][source]['all'][metric]
            row['read_benefit_pp']=100*(row['all_normal_accuracy']-row['all_disconnected_accuracy'])
            flat.append(row)
            for name in ('results.json','validation.csv','trials.csv'):
                p=directory/name
                hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        # Baselines receive identical streams under each router within a seed.
        assert reports[seed,'interaction']['budget']==reports[seed,'additive']['budget']
        for v in ('additive','concat'):
            for c in CONDITIONS:
                for source in ('bayes','marginal'):
                    assert reports[seed,v]['test'][c][source]==reports[seed,'interaction']['test'][c][source]
    assert len({hashlib.sha256(repr(streams[s]).encode()).hexdigest() for s in SEEDS})==3
    aggregate={}
    for variant in VARIANTS:
        selected=[row for row in flat if row['variant']==variant]
        aggregate[variant]={}
        for name in selected[0]:
            if name in ('seed','variant','selected_step'): continue
            values=[row[name] for row in selected]
            aggregate[variant][name]=dict(mean=statistics.mean(values),sd=statistics.stdev(values),by_seed=dict(zip(SEEDS,values)))
    differences={}
    for left,right in [('interaction','additive'),('interaction','concat'),('additive','concat')]:
        differences[f'{left}_minus_{right}']={}
        for c in CONDITIONS:
            delta=[100*(reports[s,left]['test'][c]['normal']['all']['accuracy']-reports[s,right]['test'][c]['normal']['all']['accuracy']) for s in SEEDS]
            differences[f'{left}_minus_{right}'][c]=dict(mean_pp=statistics.mean(delta),by_seed_pp=dict(zip(SEEDS,delta)))
    summary=dict(date='2026-10-30',completed_runs=9,seeds=list(SEEDS),variants=list(VARIANTS),
                 test_episodes_per_run=64,test_trials_per_condition_per_run=6144,
                 statistical_unit='Three paired seeds; each seed changes model initialization and sampled streams. SD is sample SD across seeds, not a confidence interval.',
                 total_recorded_hours=sum(r['seconds'] for r in reports.values())/3600,
                 aggregate=aggregate,paired_accuracy_differences=differences,source_sha256=hashes)
    dest=ROOT/'results'
    (dest/'FULL_SWEEP_2026-10-30.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (dest/'FULL_SWEEP_2026-10-30.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=['#176b8c','#b27639','#518b57']
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for i,v in enumerate(VARIANTS):
        for j,seed in enumerate(SEEDS):
            row=next(r for r in flat if r['seed']==seed and r['variant']==v)
            for ax,key,mult in [(axes[0],'all_normal_accuracy',100),(axes[1],'all_normal_nll',1)]:
                ax.scatter(i+(j-1)*.09,row[key]*mult,c=colors[i],marker=['o','s','^'][j],s=42)
        for ax,key,mult in [(axes[0],'all_normal_accuracy',100),(axes[1],'all_normal_nll',1)]:
            mean=aggregate[v][key]['mean']*mult
            ax.hlines(mean,i-.23,i+.23,color=colors[i],linewidth=2)
    for ax in axes[:2]:
        ax.set_xticks(range(3),['Interaction','Additive','Concat']);ax.grid(axis='y',alpha=.2)
    axes[0].set(title='All-feedback test accuracy',ylabel='Accuracy (%)',ylim=(91.5,92.8))
    axes[1].set(title='All-feedback test loss',ylabel='Clean-label NLL (lower is better)')
    for j,seed in enumerate(SEEDS):
        axes[0].scatter([],[],c='#444444',marker=['o','s','^'][j],label=f'Seed {seed}')
    axes[0].legend(fontsize=8,loc='lower right')
    x=np.arange(4)
    for i,v in enumerate(VARIANTS):
        values=[aggregate[v][f'{c}_normal_accuracy']['mean']*100 for c in CONDITIONS]
        axes[2].bar(x+(i-1)*.24,values,width=.24,color=colors[i],label=v.capitalize())
    axes[2].set_xticks(x,['All','Informative\nonly','Invariant\nonly','None'])
    axes[2].set(title='Feedback retained · three-seed means',ylabel='Accuracy (%)',ylim=(0,100))
    axes[2].legend(fontsize=8,loc='lower left');axes[2].grid(axis='y',alpha=.15);axes[2].set_axisbelow(True)
    fig.suptitle('TransformerX · completed 3-router × 3-seed sweep\nDots: individual seeds; horizontal marks: means. Accuracy dot plot uses an expanded scale.',fontsize=12)
    fig.savefig(dest/'figures/full-sweep-2026-10-30.svg',metadata={'Date':None})
    if preview: fig.savefig(preview,dpi=120)
    for v in VARIANTS:
        print(v,{key:aggregate[v][key]['mean'] for key in ('all_normal_accuracy','all_normal_nll','all_disconnected_accuracy','all_marginal_accuracy','all_bayes_accuracy','read_benefit_pp')})
    print('paired',json.dumps(differences))
    print('hours',summary['total_recorded_hours'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preview',type=Path)
    main(p.parse_args().preview)
