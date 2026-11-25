"""Reports must retain actual rewards, scientific weighting, failures and logs."""
import json
from pathlib import Path
import sys

import pytest

from ..reporting import Progress, Totals, training_metrics, training_model_report
from ..run_artifacts import artifact_environment,console_log
from ..run_mosp import execute
from .helpers import fixture


def test_reward_totals_and_equal_run_weighting_are_distinct(tmp_path,capsys):
    configs=[]
    for seed,n,reward in [(1,2,1.),(2,8,8.)]:
        directory=tmp_path/str(seed);directory.mkdir()
        config=dict(model='Synthetic/Test',seed=seed,output=str(directory))
        record=dict(model=config['model'],training_seed=seed,iterations=1,tasks=n,
            total_reward=reward,mean_reward=reward/n,totals={'reward':reward},means={'reward':reward/n,'S':reward/n})
        (directory/'training_summary.json').write_text(json.dumps(record))
        configs.append(('',config))
    result=training_model_report('Synthetic/Test',configs,tmp_path/'model')
    assert result['across_runs']['total_reward']==9
    assert result['across_runs']['mean_reward']==.75  # Equal runs, not pooled .9.
    assert result['across_runs']['tasks']==10
    assert 'summary across 2 training runs' in capsys.readouterr().out


def test_progress_counts_only_completed_work(capsys):
    progress=Progress('model | run 7 | iterations',3,min_interval=0)
    progress.advance();progress.advance()
    text=capsys.readouterr().out
    assert '0/3' in text and '2/3' in text and '100.0%' not in text
    progress.advance()
    assert '3/3 (100.0%)' in capsys.readouterr().out
    with pytest.raises(ValueError):progress.advance()


def test_failed_task_remains_in_reward_and_performance_denominator():
    torch=pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    from ..rollout import collect
    from .helpers import tiny_model,ByteTokenizer
    policy=ConstrainedPolicy(tiny_model(),ByteTokenizer(),context_limit=2)
    failed=training_metrics(collect(policy,fixture(),1,8))
    assert failed['reward']==0 and failed['limit_failure']==1 and failed['protocol_failure']==0
    totals=Totals();totals.add(dict(reward=1.,limit_failure=0.));totals.add(failed)
    assert totals.record()['mean_reward']==.5 and totals.record()['means']['limit_failure']==.5


def test_native_child_output_and_stderr_are_logged_once(tmp_path):
    root=tmp_path/'run'
    with artifact_environment(root):
        execute([sys.executable,'-c',"import os; print('python output'); os.write(2,b'native stderr\\n')"],root/'stage'/'worker.log')
    for path in [root/'log.txt',root/'stage'/'log.txt',root/'stage'/'worker.log']:
        text=path.read_text()
        assert text.count('python output')==1 and text.count('native stderr')==1


def test_logging_parent_marker_and_alias_do_not_duplicate_output(tmp_path,monkeypatch):
    path=tmp_path/'log.txt';alias=tmp_path/'alias.txt';alias.symlink_to(path)
    with console_log(path,[alias]):print('once')
    assert path.read_text().count('once')==1
    monkeypatch.setenv('MOSP_CONSOLE_LOG_PATH',str(path))
    with console_log(path):print('parent captures this')
    assert 'parent captures this' not in path.read_text()


def test_real_cpu_training_reports_exact_saved_rollout_rewards(tmp_path,monkeypatch,capsys):
    torch=pytest.importorskip('torch')
    from .. import train
    from ..policy import ConstrainedPolicy
    from .helpers import tiny_model,ByteTokenizer
    model,tokenizer=tiny_model(),ByteTokenizer()
    def save_model(path,**kwargs):
        Path(path).mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),Path(path)/'tiny_cpu_test.pt')
    model.save_pretrained=save_model
    tokenizer.save_pretrained=lambda path: (Path(path)/'tokenizer.json').write_text('{}')
    monkeypatch.setattr(train,'load_policy',lambda _:ConstrainedPolicy(model,tokenizer,context_limit=50000))
    for name in ('reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU test')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda *a,**kw:None)
    dataset=tmp_path/'train.jsonl';dataset.write_text(json.dumps(fixture().record())+'\n')
    output=tmp_path/'seed-17'
    config=dict(model='Synthetic/CPUReportingTest',revision='0'*40,seed=17,context_limit=50000,
        temperature=1.,dataset=str(dataset),output=str(output),curriculum=[1],steps=2,
        group_size=4,traversal_budget=1,objective='pareto',learning_rate=.001,
        beta=0.,clip=.2,epochs_per_group=1,max_grad_norm=1.,early_stopping=False)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    train.main(path)
    tasks=[r for file in output.glob('rollout-*.json') for r in json.loads(file.read_text())]
    saved=json.loads((output/'training_summary.json').read_text())
    assert saved['tasks']==len(tasks)==8
    assert saved['total_reward']==sum(t['reward'] for t in tasks)
    assert saved['mean_reward']==saved['total_reward']/8
    text=(output/'log.txt').read_text()
    assert 'training run complete' in text and '2/2 (100.0%)' in text
    assert 'Total reward (raw)' in text and 'Complete success' in text
    assert f"{saved['total_reward']:.4f}" in text
    assert 'training run complete' in capsys.readouterr().out


def test_saved_report_reads_partial_run_without_touching_training_data(tmp_path,capsys):
    from ..reporting import saved_run_report
    root=tmp_path/'run';trained=root/'Synthetic_trained';directory=trained/'runs/seed-1'
    directory.mkdir(parents=True);(trained/'configs').mkdir()
    config=dict(model='Synthetic/Test',seed=1,output=str(directory),steps=10,group_size=2)
    configpath=trained/'configs/train_seed_1.json';configpath.write_text(json.dumps(config))
    metrics=directory/'metrics.jsonl'
    original=''.join(json.dumps(dict(mean_reward=v,generated_lengths=[3,4]))+'\n' for v in [.5,1.])+'{"incomplete"'
    metrics.write_text(original)
    (root/'run_manifest.json').write_text(json.dumps(dict(models={'Synthetic/Test':dict(trained=str(trained))})))
    saved_run_report(root)
    text=capsys.readouterr().out
    assert '2/10' in text and 'recorded tasks so far' in text and '3.0000' in text
    assert metrics.read_text()==original and json.loads(configpath.read_text())==config
    assert not (trained/'training_summary.json').exists()
    assert 'recorded tasks so far' in (root/'log.txt').read_text()


@pytest.mark.parametrize('stage',['select-validation','summarize'])
def test_stage_log_routing_uses_input_config_even_with_trailing_options(tmp_path,monkeypatch,stage):
    from .. import run_mosp as runner
    configdir=tmp_path/'trained/configs';configdir.mkdir(parents=True)
    config=configdir/'eval.json';config.write_text(json.dumps(dict(output=str(tmp_path/'comparison'))))
    selected=configdir/'not-created-yet.json'
    args=['--selected-config',str(selected)] if stage=='select-validation' else ['--no-plots']
    class Child:
        stdout=['native stage output\n']
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:Child())
    runner.execute([sys.executable,'-m',runner.MODULE+'.eval_baselines',stage,'--config',str(config),*args],tmp_path/'stage.log')
    expected=tmp_path/'trained/validation/log.txt' if stage=='select-validation' else tmp_path/'comparison/log.txt'
    assert 'native stage output' in expected.read_text()
    assert not selected.exists()


def test_master_log_captures_launcher_failure_once(tmp_path,monkeypatch,capsys):
    from .. import run_mosp as runner
    root=tmp_path/'full_runs/run_20260101120000'
    args=runner.parser().parse_args(['--mode','prepare','--output',str(root)])
    def fail(*a):
        print('before failure')
        raise ValueError('failed stage')
    monkeypatch.setattr(runner,'_launch_actual',fail)
    with pytest.raises(SystemExit) as raised:runner.launch(args,vars(args))
    assert raised.value.code==2
    text=(root/'log.txt').read_text()
    assert text.count('before failure')==1 and text.count('failed stage')==1
    assert capsys.readouterr().err.count('failed stage')==1


def test_comparison_console_uses_weighted_estimates_and_keeps_raw_totals(monkeypatch):
    from .. import reporting
    from ..eval_baselines import METRICS
    captured=[]
    monkeypatch.setattr(reporting,'table',lambda title,headers,rows:captured.append((title,headers,rows)))
    config=dict(initial_checkpoint={'model':'Synthetic/Test'},training_runs=[{'id':'r1'},{'id':'r2'}])
    comparison=[]
    for scope in ('in_distribution','distribution_shift_macro'):
        for entity,after in [('r1',.3),('r2',.5),('training_procedure',.4)]:
            for metric in METRICS:
                comparison.append(dict(scope=scope,difficulty='all',k='all',budget_multiplier=8,
                    mode='independent_exploration',reference='initial',comparison=entity,metric=metric,
                    baseline=.2,post=after,difference=after-.2,ci_low=.05,ci_high=.15))
    rows=[dict(model_id=m,domain=d,budget_multiplier=8,mode='independent_exploration',
        metrics={key:.9 for key in METRICS}) for m in ('initial','r1','r2') for d in ('in_distribution','size')]
    reporting.comparison_report(config,rows,comparison)
    main=next(rows for title,_,rows in captured if 'held-out in_distribution' in title)
    weighted=next(r for r in main if r[0]=='Mean reward (weighted)')
    raw=next(r for r in main if r[0]=='Total reward (raw)')
    assert weighted[1]=='0.2000' and weighted[-1]=='0.4000'
    assert raw[1]=='0.9000' and raw[-1]=='1.8000'
    assert any('distribution_shift_macro' in title for title,_,_ in captured)


@pytest.mark.parametrize('low,high,assessment',[(.02,.10,'CI above zero'),(-.02,.10,'CI includes zero'),(-.10,-.02,'CI below zero')])
def test_primary_headline_uses_paired_estimates_and_separates_diagnostic(low,high,assessment,capsys):
    from ..reporting import primary_comparison_report
    config=dict(initial_checkpoint={'model':'Synthetic/Test'},training_runs=[{'id':'r1'},{'id':'r2'}])
    row=dict(scope='in_distribution',comparison='training_procedure',reference='initial',
        mode='independent_exploration',budget_multiplier=8,difficulty='all',k='all',metric='S',
        baseline=.25,post=.30,difference=.05,ci_low=low,ci_high=high)
    primary_comparison_report(config,[row,dict(row,mode='shared_transcript_selection',post=.99,difference=.74,ci_low=.70,ci_high=.80)])
    text=capsys.readouterr().out
    assert 'TRAINED vs FROZEN BASE' in text
    assert 'Frozen base S' in text and 'GRPO-trained S' in text
    assert '25.00%' in text and '30.00%' in text and '+5.00' in text
    independent,shared = text.split(' | PRIMARY B=8k | shared transcript selection',1)
    assert '99.00%' not in independent and assessment in independent
    assert '99.00%' in shared and '+74.00' in shared and '[+70.00, +80.00]' in shared
    assert '2 independent training runs' in text


def test_summary_cli_exports_comparison_headline_to_console_and_log(tmp_path,monkeypatch,capsys):
    from .. import eval_baselines as harness
    from ..reporting import primary_comparison_report
    output=tmp_path/'comparison'
    config=dict(output=str(output),initial_checkpoint={'model':'Synthetic/Test'},training_runs=[{'id':'r1'}])
    path=tmp_path/'evaluation.json';path.write_text(json.dumps(config))
    row=dict(scope='in_distribution',comparison='training_procedure',reference='initial',
        mode='independent_exploration',budget_multiplier=8,difficulty='all',k='all',metric='S',
        baseline=.2,post=.4,difference=.2,ci_low=.1,ci_high=.3)
    monkeypatch.setattr(harness,'validate_config',lambda _:None)
    monkeypatch.setattr(harness,'summarize',lambda c,o,p:primary_comparison_report(c,[row]))
    monkeypatch.setattr(sys,'argv',['eval_baselines','summarize','--config',str(path),'--no-plots'])
    harness.main()
    console=capsys.readouterr().out
    log=(output/'log.txt').read_text()
    assert console==log
    assert 'TRAINED vs FROZEN BASE' in log and '20.00%' in log and '40.00%' in log
    assert '+20.00' in log and '[+10.00, +30.00]' in log
