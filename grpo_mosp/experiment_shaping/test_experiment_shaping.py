"""The seven-agent entry point shares data, one frozen reference and one report."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from . import experiment_shaping as experiment
from .. import run_mosp_shaping as runner
from .. import eval_shaping as evaluation
from ..shaping import SIGNALS


def test_default_plan_has_exactly_seven_agents_and_one_frozen_reference(tmp_path):
    output=tmp_path/'full_runs/run_20260101120000'
    args=experiment.configure(experiment.parser().parse_args(['--dry-run','--output',str(output)]))
    root,settings,_=runner.resolve(args)
    _,training,config=runner.configurations(root,settings)
    evaluation.validate_config(config)
    assert config['conditions']==list(experiment.COMBINATIONS)
    assert len(training)==7 and len(config['model_output_dirs'])==8
    assert list(config['model_output_dirs']).count('initial')==1
    assert len({cfg['dataset'] for _,cfg in training})==1
    assert len({cfg['validation_dataset'] for _,cfg in training})==1
    assert len({json.dumps(cfg['initial_checkpoint'],sort_keys=True) for _,cfg in training})==1
    for key in ('shuffle_graphs','balance_graphs','adaptive_graph_sampling'):assert settings[key]
    all_config=next(cfg for _,cfg in training if cfg['shaping_condition']=='all')
    assert all(all_config['shaping'][name] for name in SIGNALS)
    assert all_config['steps']==32 and all_config['group_size']==4 and all_config['traversal_budget']==16
    assert config['ks']==[2] and config['graphs']==10 and len(config['modes'])==2
    assert not root.exists()


def test_profile_overrides_and_conflicting_selectors():
    args=experiment.configure(experiment.parser().parse_args(['--dry-run','--steps','6',
        '--training-seeds','17,29','--attempt-multiplier','4']))
    _,settings,_=runner.resolve(args)
    assert settings['steps']==6 and settings['training_runs']==2
    assert settings['training_seeds']==[17,29] and settings['attempts_per_graph'] is None
    assert settings['attempt_multiplier']==4
    for flags in (['--shaping','factorial'],['--shaping-edge','true']):
        with pytest.raises(ValueError,match='fixes the seven'):
            experiment.configure(experiment.parser().parse_args(flags))
    args=runner.parser().parse_args(['--dry-run','--shaping-combinations','all','+'.join(SIGNALS)])
    with pytest.raises(ValueError,match='Duplicate'):runner.resolve(args)


def test_shell_prepare_and_saved_configuration_reuse(tmp_path):
    output=tmp_path/'full_runs/run_20260101120000'
    script=Path(experiment.__file__).with_suffix('.sh')
    command=[str(script),'--python',sys.executable,'--mode','prepare','--output',str(output),
        '--train-graphs','3','--validation-graphs','1','--steps','6','--curriculum-k','1',
        '--eval-ks','1','--max-paths','128','--difficulty-cutoff','4']
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    saved=list(output.glob('*_trained/shaping/launch_config.json'))[0]
    settings=json.loads(saved.read_text())
    configs=list(saved.parent.glob('configs/*_seed_*.json'))
    assert len(configs)==7 and settings['steps']==6 and settings['eval_ks']==[1]
    result=subprocess.run([str(script),'--python',sys.executable,'--mode','eval',
        '--output',str(output),'--dry-run'],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    plan=json.loads(result.stdout)
    assert plan['settings']==settings and len(plan['training_configs'])==7
    assert plan['evaluation_config']['conditions']==list(experiment.COMBINATIONS)
    assert (output/'log.txt').read_text().splitlines()[-1].startswith('TOTAL PIPELINE ELAPSED TIME:')


def test_combined_cpu_evaluation_loads_frozen_once_and_reports_all_conditions(tmp_path):
    from ..policy import ConstrainedPolicy
    from ..tests.helpers import tiny_model,ByteTokenizer
    root=tmp_path/'full_runs/run_20260101120000'
    args=experiment.configure(experiment.parser().parse_args(['--mode','prepare','--output',str(root),
        '--model','Synthetic/Seven','--revision','0'*40,'--backend','transformers',
        '--train-graphs','3','--validation-graphs','1','--steps','6','--curriculum-k','1',
        '--eval-ks','1','--max-paths','128','--difficulty-cutoff','4','--bootstrap-samples','3',
        '--max-task-tokens','64']))
    root,settings,_=runner.resolve(args);root.mkdir(parents=True)
    _,training,config=runner.configurations(root,settings)
    runner.graphs.make_data(root,settings)
    config['device']='cpu'
    for path,cfg in training:
        directory=Path(cfg['output']);(directory/'adapter').mkdir(parents=True)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(cfg));(directory/'config.json').write_text(json.dumps(cfg))
        (directory/'adapter/weights.bin').write_bytes(str(path).encode())
        (directory/'training_state.pt').write_bytes(b'Synthetic fixture, not a learned checkpoint')
        (directory/'training_summary.json').write_text('{}')
        (directory/'early_stopping.json').write_text(json.dumps(dict(enabled=False,selected_step=6)))
        (directory/'training_compute.json').write_text(json.dumps(dict(kind='CPU fixture')))
    output=Path(config['output']);evaluation.prepare(config,output)
    loaded=[]
    def loader(spec,controls):
        loaded.append(spec)
        return ConstrainedPolicy(tiny_model(),ByteTokenizer(),controls['context_limit'])
    evaluation.run(config,output,loader)
    rows,_=evaluation.checked_results(config,output)
    assert len(loaded)==8 and sum('adapter' not in spec for spec in loaded)==1
    assert len(rows)==160 and len([r for r in rows if r['model_id']=='initial'])==20
    records=evaluation.summarize(config,output,plots=True)
    assert all(any(r['condition']==c and r['reference']=='initial' and r['metric']=='S'
                   for r in records) for c in experiment.COMBINATIONS)
    assert any(r['condition']=='all' and r['reference']=='none' for r in records)
    for mode in config['modes']:
        assert (root/'figures/Seven/evaluation'/mode/'checkpoint_comparison/reward.png').exists()
    text=(output/'comparison.md').read_text()
    assert all(c in text for c in experiment.COMBINATIONS)
    evaluation.run(config,output,loader)
    assert len(loaded)==8  # Completed cells are reused, including the frozen base.
