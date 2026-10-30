import json
import torch
from transformerx.model import Config, StreamModel
from transformerx.silent_data import RuleTokenizer
from transformerx.silent import Options, evaluate, train, load_model


def test_prediction_precedes_feedback_and_interventions_are_paired():
    torch.manual_seed(3)
    model = StreamModel(Config(dim=16,state_dim=8,layers=1,rank=2,num_classes=2),
                        tokenizer=RuleTokenizer())
    opts = Options(trials=8,batch=2,eval_episodes=2,test_episodes=2)
    a = evaluate(model,opts,episodes=2,feedback='all')
    b = evaluate(model,opts,episodes=2,feedback='none')
    for i in (0,8):
        assert a['trials'][i]['p_right'] == b['trials'][i]['p_right']
    assert a['restoration_max_error'] == 0
    assert a['normal']['all']['count'] == 16
    assert a['bayes']['all']['count'] == 16
    assert a['disconnected']['all']['count'] == 16


def test_checkpoint_resume_and_evaluation(tmp_path):
    opts=Options(steps=1,batch=2,trials=6,tbptt=3,eval_every=1,
                 eval_episodes=2,test_episodes=2,threads=1,out=str(tmp_path))
    cfg=Config(dim=16,state_dim=8,layers=1,rank=2,num_classes=2,router='interaction')
    result=train(cfg,opts)
    assert result['completed_steps']==1
    assert set(result['test'])=={'all','informative_only','invariant_only','none'}
    model,payload=load_model(tmp_path/'latest.pt')
    assert isinstance(model.tokenizer,RuleTokenizer)
    opts.steps=2
    result=train(cfg,opts,resume=tmp_path/'latest.pt')
    assert result['completed_steps']==2
    assert json.loads((tmp_path/'results.json').read_text())['selected_step'] in (1,2)
    assert (tmp_path/'trials.csv').exists()


def test_resume_matches_uninterrupted_training(tmp_path):
    from dataclasses import replace
    cfg=Config(dim=16,state_dim=8,layers=1,rank=2,num_classes=2,router='interaction')
    opts=Options(steps=2,batch=2,trials=4,tbptt=2,eval_every=1,
                 eval_episodes=1,test_episodes=1,threads=1,out=str(tmp_path/'whole'))
    train(cfg,opts)
    partial=replace(opts,steps=1,out=str(tmp_path/'partial'))
    train(cfg,partial)
    partial.steps=2
    train(cfg,partial,resume=tmp_path/'partial/latest.pt')
    _,a=load_model(tmp_path/'whole/latest.pt')
    _,b=load_model(tmp_path/'partial/latest.pt')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


def test_standalone_evaluation_preserves_training_report_and_records_settings(tmp_path):
    from transformerx.silent import main
    cfg=Config(dim=16,state_dim=8,layers=1,rank=2,num_classes=2,router='interaction')
    opts=Options(steps=1,batch=1,trials=3,tbptt=3,eval_every=1,
                 eval_episodes=1,test_episodes=1,threads=1,out=str(tmp_path))
    train(cfg,opts)
    original=(tmp_path/'trials.csv').read_bytes()
    main(['--evaluate',str(tmp_path/'best.pt'),'--hazard','1.0'])
    assert (tmp_path/'trials.csv').read_bytes()==original
    report=json.loads((tmp_path/'evaluation.json').read_text())
    assert report['evaluation']['hazard']==1.0
    assert report['checkpoint'].endswith('best.pt')
    assert report['model']['router']=='interaction'
    assert (tmp_path/'evaluation_trials.csv').exists()
