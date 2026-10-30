"""Matched sequential GPU jobs for silent rules; dry-run before a large sweep."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys

VARIANTS = {'loop-interaction':('loop','interaction'), 'loop-additive':('loop','additive'),
            'loop-concat':('loop','concat'), 'gru':('gru','concat'),
            'tokens':('tokens','concat'), 'none':('none','concat'), 'window':('window','concat')}


def commands(out,seeds,variants,backbone,steps,device,amp,batch=None,tbptt=None):
    if len(set(seeds)) != len(seeds) or len(set(variants)) != len(variants):
        raise ValueError("Duplicate seeds or variants would reuse output directories")
    jobs=[]
    for seed in seeds:
        for variant in variants:
            if variant not in VARIANTS: raise ValueError(f'Unknown variant: {variant}')
            mode,router=VARIANTS[variant]
            job=[sys.executable,'-m','transformerx.silent','--mode',mode,'--router',router,
                 '--backbone',backbone,'--steps',str(steps),'--seed',str(seed),'--device',device,
                 '--out',str(Path(out)/variant/f'seed-{seed}')]
            if amp: job.append('--amp')
            if batch is not None: job += ['--batch',str(batch)]
            if tbptt is not None: job += ['--tbptt',str(tbptt)]
            jobs.append(job)
    return jobs


def summarize(out, jobs):
    rows=[]
    signatures=set()
    for job in jobs:
        path=Path(job[job.index('--out')+1])/'results.json'
        report=json.loads(path.read_text())
        # Same data/optimization and backbone settings; model capacity is reported,
        # not claimed to be matched across unlike baseline architectures.
        cfg=report['model']; opt=report['training']
        for flag,actual in (('--mode',cfg['mode']),('--router',cfg['router']),
                            ('--backbone',cfg['backbone']),('--seed',opt['seed']),
                            ('--steps',report['completed_steps']),('--batch',opt['batch']),
                            ('--tbptt',opt['tbptt'])):
            if flag in job and str(actual)!=job[job.index(flag)+1]:
                raise ValueError(f'Report does not match requested job: {path}, {flag}')
        if opt['amp'] != ('--amp' in job):
            raise ValueError(f'Report does not match requested AMP setting: {path}')
        signatures.add(json.dumps({k:v for k,v in opt.items() if k not in ('seed','out','device','threads')},sort_keys=True)
                       +json.dumps({k:v for k,v in cfg.items() if k not in ('mode','router')},sort_keys=True))
        row=dict(variant=path.parent.parent.name,seed=opt['seed'],selected_step=report['selected_step'],
                 **report['budget'])
        for condition,metrics in report['test'].items():
            for source in ('normal','bayes','marginal','disconnected','reset'):
                for category in ('all','informative','invariant','returning_rule'):
                    if category in metrics[source]:
                        row[f'{condition}_{source}_{category}_accuracy']=metrics[source][category]['accuracy']
        rows.append(row)
    if len(signatures)>1: raise ValueError('Refusing to summarize unmatched configurations')
    dest=Path(out)/'summary.csv'
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with dest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/silent-suite')
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--backbone',choices=['tiny','hf'],default='tiny')
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--batch',type=int);p.add_argument('--tbptt',type=int)
    p.add_argument('--device',default='auto');p.add_argument('--amp',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    jobs=commands(args.out,args.seeds,args.variants,args.backbone,args.steps,args.device,args.amp,args.batch,args.tbptt)
    for job in jobs:
        run=list(job)
        latest=Path(job[job.index('--out')+1])/'latest.pt'
        if args.resume and latest.exists(): run+=['--resume',str(latest)]
        if args.dry_run: print(shlex.join(run))
        elif not args.summary_only: subprocess.run(run,check=True)
    if not args.dry_run: summarize(args.out,jobs)


if __name__=='__main__': main()
