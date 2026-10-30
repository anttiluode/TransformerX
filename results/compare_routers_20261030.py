"""Audit completed seed-7 router results and bootstrap paired test episodes.
Run: python results/compare_routers_20261030.py (requires NumPy).
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    variants=('interaction','additive')
    reports={}; tables={}; hashes={}
    for variant in variants:
        directory=ROOT/'runs/silent-routing'/f'loop-{variant}'/'seed-7'
        reports[variant]=json.loads((directory/'results.json').read_text())
        rows=list(csv.DictReader((directory/'trials.csv').open()))
        keys=[(r['feedback'],int(r['episode']),int(r['trial'])) for r in rows]
        assert len(keys)==len(set(keys))==24576
        tables[variant]=dict(zip(keys,rows))
        for filename in ('results.json','trials.csv','validation.csv'):
            path=directory/filename
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for condition,metrics in reports[variant]['test'].items():
            subset=[r for r in rows if r['feedback']==condition]
            assert len(subset)==6144
            y=np.array([int(r['truth']) for r in subset])
            for source,col in [('normal','p_right'),('disconnected','disconnected_p_right'),
                               ('bayes','bayes_p_right'),('reset','reset_p_right')]:
                p=np.array([float(r[col]) for r in subset])
                assert np.isclose(np.mean((p>.5)==y),metrics[source]['all']['accuracy'])
                nll=-np.log(np.maximum(np.where(y,p,1-p),1e-12)).mean()
                assert np.isclose(nll,metrics[source]['all']['nll'],atol=1e-6)
    a,b=[reports[v] for v in variants]
    assert {k:v for k,v in a['model'].items() if k!='router'}=={k:v for k,v in b['model'].items() if k!='router'}
    assert {k:v for k,v in a['training'].items() if k!='out'}=={k:v for k,v in b['training'].items() if k!='out'}
    assert a['budget']==b['budget']
    assert tables['interaction'].keys()==tables['additive'].keys()
    for key,r in tables['interaction'].items():
        other=tables['additive'][key]
        for field in ('tone','rule','truth','outcome','informative','switched','age','returning'):
            assert r[field]==other[field],(key,field)
    comparisons={}
    rng=np.random.default_rng(20261030)
    bootstrap_indices=rng.integers(0,64,(10000,64))
    for condition in a['test']:
        correct={}
        for variant in variants:
            correct[variant]=np.array([
                [(float(tables[variant][condition,e,t]['p_right'])>.5)==int(tables[variant][condition,e,t]['truth'])
                 for t in range(96)] for e in range(64)])
        x,y=[correct[v] for v in variants]
        delta=x.astype(float)-y.astype(float)
        boot=delta.mean(axis=1)[bootstrap_indices].mean(axis=1)*100
        comparisons[condition]=dict(interaction_correct=int(x.sum()),additive_correct=int(y.sum()),
            interaction_only_correct=int((x&~y).sum()),additive_only_correct=int((y&~x).sum()),
            interaction_minus_additive_pp=float(delta.mean()*100),
            paired_episode_bootstrap_95_percentile_pp=np.quantile(boot,[.025,.975]).tolist(),
            interaction_minus_additive_nll=a['test'][condition]['normal']['all']['nll']-b['test'][condition]['normal']['all']['nll'])
    result=dict(date='2026-10-30',training_seed=7,trials_per_condition=6144,episodes=64,
                bootstrap_resamples=10000,bootstrap_seed=20261030,
                scope='Test-episode uncertainty conditional on two selected seed-7 checkpoints; not training-seed uncertainty or an equivalence test',
                selected_steps={v:reports[v]['selected_step'] for v in variants},
                matched_data_and_configuration=True,comparisons=comparisons,source_sha256=hashes)
    (ROOT/'results/ROUTER_COMPARISON_2026-10-30.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
