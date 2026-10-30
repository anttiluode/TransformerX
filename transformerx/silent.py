"""Outcome-driven hidden-rule benchmark: python -m transformerx.silent --help."""
import argparse
import csv
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, StreamModel
from .run import (atomic_json, device_for, model_weights, restore_weights,
                  optimizer_step)
from .silent_data import BayesFilter, RuleTokenizer, generate, feedback_visible

FEEDBACK = ('all', 'informative_only', 'invariant_only', 'none')


@dataclass
class Options:
    steps: int = 500
    batch: int = 8
    trials: int = 96
    hazard: float = .05
    feedback_noise: float = .05
    informative_fraction: float = .6
    lr: float = .001
    seed: int = 7
    tbptt: int = 16
    eval_every: int = 25
    eval_episodes: int = 32
    test_episodes: int = 64
    device: str = 'auto'
    amp: bool = False
    threads: int = 4
    out: str = 'runs/silent'


def episodes_for(opts, split, start, count):
    return [generate(split, opts.seed * 10_000_000 + start + i, opts.trials,
                     opts.hazard, opts.feedback_noise, opts.informative_fraction)
            for i in range(count)]


def validate(cfg, opts):
    if cfg.task != 'language' or cfg.num_classes != 2:
        raise ValueError('Silent rules require language observations and two answer classes')
    if min(opts.steps,opts.batch,opts.trials,opts.tbptt,opts.eval_every,
           opts.eval_episodes,opts.test_episodes,opts.threads) < 1 or opts.lr <= 0:
        raise ValueError('Counts and learning rate must be positive')
    episodes_for(opts, 'val', 0, 1)  # validates the generative process too


def add_score(groups, trial, probs):
    age = str(trial.age) if trial.age < 5 else '5-9' if trial.age < 10 else '10+'
    labels = ['all', 'informative' if trial.informative else 'invariant',
              'age_'+age, 'returning_rule' if trial.returning else 'first_visit']
    y = trial.truth
    for label in labels:
        row = groups.setdefault(label, {'count':0,'accuracy':0.,'nll':0.,'brier':0.})
        row['count'] += 1
        row['accuracy'] += int(np.argmax(probs) == y)
        row['nll'] -= math.log(max(float(probs[y]), 1e-12))
        row['brier'] += sum((float(p)-int(j==y))**2 for j,p in enumerate(probs))


@torch.inference_mode()
def evaluate(model, opts, split='val', episodes=None, feedback='all', paired=True):
    if feedback not in FEEDBACK:
        raise ValueError('Unknown feedback condition')
    model.eval()
    count_episodes = episodes if episodes is not None else opts.eval_episodes
    scores = {k:{} for k in ('normal','bayes','marginal')}
    if paired:
        scores.update(disconnected={}, reset={})
    rows, restoration, rule_correct, total = [], 0., 0, 0
    for start in range(0,count_episodes,opts.batch):
        batch = episodes_for(opts,split,start,min(opts.batch,count_episodes-start))
        state = model.initial_state(len(batch))
        filters = [BayesFilter(opts.hazard,opts.feedback_noise) for _ in batch]
        for t in range(opts.trials):
            trials = [ep[t] for ep in batch]
            obs = [trial.query for trial in trials]
            logits, next_state, diag = model.forward_step(obs,state)
            probabilities = {'normal':logits.float().softmax(-1).cpu().numpy()}
            if paired:
                off, _, _ = model.forward_step(obs,state,context_enabled=False,update_state=False)
                restored, _, _ = model.forward_step(obs,state,update_state=False)
                reset, _, _ = model.forward_step(obs,model.initial_state(len(batch)))
                restoration = max(restoration,(restored-logits).abs().max().item())
                probabilities['disconnected'] = off.float().softmax(-1).cpu().numpy()
                probabilities['reset'] = reset.float().softmax(-1).cpu().numpy()
            for i,trial in enumerate(trials):
                belief = filters[i]
                if t:
                    belief.advance()
                oracle = belief.predict(trial.tone)
                marginal = BayesFilter(opts.hazard,opts.feedback_noise).predict(trial.tone)
                for name,probs in probabilities.items():
                    add_score(scores[name],trial,probs[i])
                add_score(scores['bayes'],trial,oracle)
                add_score(scores['marginal'],trial,marginal)
                rule_correct += int(np.argmax(belief.belief)==trial.rule)
                total += 1
                row = dict(episode=start+i,trial=t,tone=trial.tone,rule=trial.rule,
                           truth=trial.truth,outcome=trial.outcome,informative=trial.informative,
                           switched=trial.switched,age=trial.age,returning=trial.returning,
                           p_right=float(probabilities['normal'][i,1]),
                           bayes_p_right=float(oracle[1]))
                if paired:
                    row.update(disconnected_p_right=float(probabilities['disconnected'][i,1]),
                               reset_p_right=float(probabilities['reset'][i,1]))
                if 'route_weights' in diag:
                    row.update({f'adapter_{k}':float(p) for k,p in enumerate(diag['route_weights'][i])})
                rows.append(row)
                # Current feedback is consumed only AFTER all predictions/scoring.
                if feedback_visible(trial,feedback):
                    belief.observe(trial.tone,trial.outcome)
            texts = [tr.feedback if feedback_visible(tr,feedback) else 'no feedback' for tr in trials]
            _, state, _ = model.forward_step(texts,next_state)
    for groups in scores.values():
        for row in groups.values():
            for key in ('accuracy','nll','brier'):
                row[key] /= row['count']
    return {**scores, 'bayes_rule_accuracy':rule_correct/total,
            'restoration_max_error':restoration if paired else None,
            'trials':sorted(rows,key=lambda r:(r['episode'],r['trial']))}


def save(path, model, optimizer, scaler, opts, step, best):
    payload = dict(experiment='silent_rules_v1', model_config=asdict(model.cfg),
                   options=asdict(opts), model=model_weights(model), optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(),step=step,best_val_nll=best,rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    tmp = Path(path).with_suffix('.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)


def read(path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('experiment') != 'silent_rules_v1':
        raise ValueError('Expected a silent-rule checkpoint')
    return payload


def load_model(path,device='cpu'):
    payload = read(path)
    model = StreamModel(Config(**payload['model_config']),tokenizer=RuleTokenizer()).to(device_for(str(device)))
    restore_weights(model,payload)
    return model.eval(),payload


def final_evaluation(model,opts,trial_filename='trials.csv'):
    results, rows = {}, []
    for condition in FEEDBACK:
        result = evaluate(model,opts,'test',opts.test_episodes,condition)
        rows.extend(dict(feedback=condition,**r) for r in result.pop('trials'))
        results[condition] = result
    dest = Path(opts.out)
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/trial_filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return results


def train(cfg,opts,resume=None):
    validate(cfg,opts)
    torch.set_num_threads(opts.threads)
    torch.manual_seed(opts.seed)
    device=device_for(opts.device)
    out=Path(opts.out); out.mkdir(parents=True,exist_ok=True)
    if not resume and (out/'latest.pt').exists():
        raise ValueError('Checkpoint exists; use --resume or a fresh output directory')
    model=StreamModel(cfg,tokenizer=RuleTokenizer()).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=opts.lr)
    amp=opts.amp and device.type=='cuda'
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    first,best=0,float('inf')
    if resume:
        payload=read(resume)
        if payload['model_config'] != asdict(cfg):
            raise ValueError('Resume model configuration differs')
        mutable={'steps','out','device','threads','test_episodes'}
        for name,value in asdict(opts).items():
            if name not in mutable and payload['options'][name]!=value:
                raise ValueError(f'Resume configuration differs: {name}')
        restore_weights(model,payload); optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        first,best=payload['step'],payload['best_val_nll']
        torch.set_rng_state(payload['rng'])
        if device.type=='cuda' and payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        if opts.steps<first:
            raise ValueError('--steps cannot precede checkpoint step')
        source=Path(resume).parent/'best.pt'
        if source.resolve()!=(out/'best.pt').resolve():
            shutil.copy2(source,out/'best.pt')
    atomic_json(out/'config.json',dict(model=asdict(cfg),training=asdict(opts),budget=model.budget()))
    started=time.perf_counter()
    for step in range(first,opts.steps):
        model.train()
        batch=episodes_for(opts,'train',step*opts.batch,opts.batch)
        state=model.initial_state(opts.batch)
        optimizer.zero_grad(set_to_none=True)
        losses=[]; train_nll=0.
        for t in range(opts.trials):
            with torch.autocast(device.type,dtype=dtype,enabled=amp):
                logits,state,_=model.forward_step([ep[t].query for ep in batch],state)
                # Learn only from the feedback actually available to the agent.
                target=torch.tensor([ep[t].outcome for ep in batch],device=device)
                loss=F.cross_entropy(logits.float(),target)
                losses.append(loss); train_nll+=loss.detach().item()
                _,state,_=model.forward_step([ep[t].feedback for ep in batch],state)
            if (t+1)%opts.tbptt==0 or t+1==opts.trials:
                objective=torch.stack(losses).mean()
                if not torch.isfinite(objective):
                    raise FloatingPointError('Nonfinite training loss')
                optimizer_step(objective,optimizer,scaler,model.parameters())
                state=model.detach_state(state)
                optimizer.zero_grad(set_to_none=True); losses=[]
        if (step+1)%opts.eval_every==0 or step+1==opts.steps:
            val=evaluate(model,opts,paired=False)['normal']['all']
            if val['nll']<best:
                best=val['nll']; save(out/'best.pt',model,optimizer,scaler,opts,step+1,best)
            save(out/'latest.pt',model,optimizer,scaler,opts,step+1,best)
            log=out/'validation.csv'; exists=log.exists()
            with log.open('a',newline='') as f:
                w=csv.writer(f)
                if not exists: w.writerow(['step','train_noisy_nll','val_clean_nll','val_accuracy'])
                w.writerow([step+1,train_nll/opts.trials,val['nll'],val['accuracy']])
            print(f'{cfg.mode}/{cfg.router} {step+1}/{opts.steps}: val NLL {val["nll"]:.4f}, accuracy {val["accuracy"]:.3f}',flush=True)
    selected,payload=load_model(out/'best.pt',str(device))
    report=dict(completed_steps=opts.steps,selected_step=payload['step'],model=asdict(cfg),
                training=asdict(opts),budget=model.budget(),test=final_evaluation(selected,opts),
                seconds=time.perf_counter()-started,
                status='Synthetic diagnostic; no claim of preventing catastrophic forgetting',
                environment=dict(torch=str(torch.__version__),device=str(device),
                                 gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None))
    atomic_json(out/'results.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume'); p.add_argument('--evaluate')
    for cls in (Config,Options):
        for field in fields(cls):
            kwargs=dict(default=argparse.SUPPRESS)
            if field.type is bool: kwargs['action']=argparse.BooleanOptionalAction
            else: kwargs['type']=field.type
            p.add_argument('--'+field.name.replace('_','-'),**kwargs)
    args=vars(p.parse_args(argv)); resume=args.pop('resume'); checkpoint=args.pop('evaluate')
    if resume and checkpoint: raise ValueError('Choose resume or evaluate')
    defaults={'num_classes':2,'router':'interaction'}
    training={}
    if resume or checkpoint:
        payload=read(resume or checkpoint)
        defaults=payload['model_config']; training=payload['options']
        training['out']=str(Path(resume or checkpoint).parent)
    names={f.name for f in fields(Config)}
    cfg=Config(**(defaults|{k:v for k,v in args.items() if k in names}))
    opts=Options(**(training|{k:v for k,v in args.items() if k not in names}))
    validate(cfg,opts)
    if checkpoint:
        if asdict(cfg)!=payload['model_config']: raise ValueError('Cannot change checkpoint architecture')
        torch.set_num_threads(opts.threads)
        model,_=load_model(checkpoint,opts.device)
        result=final_evaluation(model,opts,trial_filename='evaluation_trials.csv')
        atomic_json(Path(opts.out)/'evaluation.json',dict(
            checkpoint=str(Path(checkpoint).resolve()),selected_step=payload['step'],
            model=asdict(cfg),evaluation=asdict(opts),test=result))
    else:
        train(cfg,opts,resume)
    print(f'Results saved to {opts.out}')


if __name__=='__main__':
    main()
