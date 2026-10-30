import pytest
from transformerx.silent_suite import commands


def test_matched_commands_and_seed_pairing():
    jobs=commands('runs/test',[7,17],['loop-interaction','loop-additive','gru'],'tiny',100,'cpu',False)
    assert len(jobs)==6
    for cmd in jobs:
        assert cmd[cmd.index('--steps')+1]=='100'
        assert cmd[cmd.index('--seed')+1] in ('7','17')
    first,second=jobs[0],jobs[1]
    assert first[first.index('--mode')+1]==second[second.index('--mode')+1]=='loop'
    assert first[first.index('--router')+1]=='interaction'
    assert second[second.index('--router')+1]=='additive'
    with pytest.raises(ValueError): commands('runs/test',[7],['bad'],'tiny',1,'cpu',False)


def test_suite_passes_gpu_memory_controls():
    job=commands('runs/test',[7],['loop-interaction'],'hf',10,'cuda',True,batch=2,tbptt=8)[0]
    assert job[job.index('--batch')+1]=='2'
    assert job[job.index('--tbptt')+1]=='8'


def test_summary_rejects_reports_that_do_not_match_requested_job(tmp_path):
    import json
    from dataclasses import asdict
    from pathlib import Path
    from transformerx.model import Config
    from transformerx.silent import Options
    from transformerx.silent_suite import summarize
    jobs=commands(str(tmp_path),[7],['loop-interaction'],'hf',10,'cpu',False)
    dest=Path(jobs[0][jobs[0].index('--out')+1]);dest.mkdir(parents=True)
    report=dict(model=asdict(Config(num_classes=2,router='interaction',backbone='tiny')),
                training=asdict(Options(steps=10)),selected_step=10,completed_steps=10,budget={},test={})
    (dest/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='requested'):
        summarize(tmp_path,jobs)
