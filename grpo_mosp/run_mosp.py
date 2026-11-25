"""Configuration and orchestration behind run_mosp.sh; no shell interpolation.

--dry-run uses only the standard library and writes nothing. Actual stages defer
imports so --mode prepare works with the CPU dataset dependencies alone.
"""
import time
PROCESS_STARTED_AT = time.time()

import argparse
from .model_options import add_thinking_arguments, thinking_options
from .training_options import TRAINING_DEFAULTS, add_training_arguments, training_options
from collections import Counter
import importlib.util
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODULE = 'jm_test.graph_world.grpo_mosp'
from .model_defaults import DEFAULT_MODEL, DEFAULT_REVISION, initial_checkpoint, loading_contract
from .run_artifacts import BASELINE_PRESET_DIR, artifact_environment, model_slug, new_run_directory, validate_run_directory, RUN_TIMEZONE
from .pipeline_timing import PipelineTiming, timed_component
from .reporting import Progress, training_model_report, training_run_record, task_tables
from .graph_sampling import SAMPLING_DEFAULTS, add_sampling_arguments, sampling_options
DEFAULTS = dict(
    **SAMPLING_DEFAULTS,
    model=DEFAULT_MODEL, revision=DEFAULT_REVISION, backend='auto', compute_dtype='bfloat16',
    offload_embedding=True, lora_alpha=16, lora_last_layers=8, lora_mlp=False,
    min_free_gpu_gib=24., max_gpu_memory_gib=38.,
    train_graphs=32, validation_graphs=32, eval_graphs=200,
    subgraphs=[2, 2], nodes_per_subgraph=[6, 12], outdegree=[2, 2],
    large_nodes_per_subgraph=None, density_outdegree=None, objectives=3,
    max_paths=4096, max_candidates=200000, difficulty_cutoff=32,
    training_runs=5, training_seeds=None, training_seed=101, seed_stride=101,
    data_seed=42, suite_seed=2000000, eval_seed=112358, bootstrap_seed=271828,
    curriculum=[1, 2, 3], eval_ks=None, attempts_per_graph=None, attempt_multiplier=8,
    **TRAINING_DEFAULTS, group_size=8, learning_rate=1e-5,
    beta=.01, clip=.2, max_grad_norm=1., lora_rank=8, objective='pareto', pareto_solver='exhaustive',
    reasoning_effort='xhigh', preserve_thinking=False,
    history_format='state_actions', temperature=1., context_limit=32768, max_action_tokens=512, max_task_tokens=16384,
    eval_repetitions=5, eval_relabelings=1, eval_budget_multipliers=[1, 2, 4, 8, 16],
    bootstrap_samples=2000,
)


def csv_ints(value):
    try:
        result = [int(v) for v in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use comma-separated integers, e.g. 1,2,3') from error
    if not result:
        raise argparse.ArgumentTypeError('At least one integer is required')
    return result


def parser(*, curriculum_flag='--curriculum'):
    p = argparse.ArgumentParser(prog='run_mosp.sh', description=__doc__, epilog=(
        'Defaults: 5 independent training runs, 32 train + 32 validation graphs, '
        '200 held-out families, 5 evaluation repetitions. Use --dry-run to inspect '
        'the complete plan. Existing output settings are loaded for later stages.'))
    add_thinking_arguments(p)
    add_sampling_arguments(p)
    add_training_arguments(p, schedule=False)
    p.add_argument('--python', help='Interpreter used by the shell launcher (or set MOSP_PYTHON)')
    p.add_argument('--mode', choices=['all', 'prepare', 'train', 'eval'], default='all',
                   help='all: data + training + comparison; prepare: data/configs only; train: data + training; eval: saved-run comparison')
    p.add_argument('--runs-root', type=Path, default=HERE/'full_runs',
                   help='Parent directory named full_runs for new timestamped runs')
    p.add_argument('--output', type=Path,
                   help='Exact full_runs/run_YYYYMMDDHHMMSS directory; use an existing one for later stages')
    p.add_argument('--dry-run', action='store_true', help='Print resolved settings, configs and commands; write nothing')
    p.add_argument('--no-plots', action='store_true', help='Skip performance plots; weighted graph images are always exported')
    def add(flag, key=None, **kw):
        key = key or flag.removeprefix('--').replace('-', '_')
        description = kw.pop('help', '')
        default = DEFAULTS[key]
        p.add_argument(flag, dest=key, default=argparse.SUPPRESS,
                       help=f'{description} [default: {default}]', **kw)
    add('--model', help='Hugging Face causal Qwen checkpoint')
    add('--backend',choices=['auto','unsloth_nf4','transformers'],help='Shared train/eval loader; auto selects NF4 for the default model')
    add('--history-format',choices=['state_actions','delta','full'],help='Full current memory plus all moves (default), lossless observation deltas, or full raw history; shared by training/evaluation')
    add('--compute-dtype',choices=['bfloat16','float16'])
    add('--offload-embedding',action=argparse.BooleanOptionalAction)
    add('--lora-mlp',action=argparse.BooleanOptionalAction)
    add('--lora-alpha',type=int)
    add('--lora-last-layers',type=int,help='Adapt only the last N language layers; 0 adapts all')
    add('--min-free-gpu-gib',type=float)
    add('--max-gpu-memory-gib',type=float,help='CUDA allocator ceiling for NF4; 0 disables')
    add('--revision', help='Full immutable checkpoint commit SHA; required when changing model')
    add('--train-graphs', type=int, help='Distinct fixed training graph instances (not GRPO group size)')
    p.add_argument('--graph-replicates', dest='train_graphs', type=int, default=argparse.SUPPRESS,
                   help='Alias for --train-graphs')
    add('--validation-graphs', type=int)
    add('--eval-graphs', type=int, help='Held-out graph families across five domains; multiple of 10')
    add('--subgraphs', nargs='+', type=int, metavar='N', help='Fixed count N, or inclusive MIN MAX; minimum 2')
    add('--nodes-per-subgraph', nargs=2, type=int, metavar=('MIN', 'MAX'), help='Inclusive node-count range for each subgraph')
    add('--outdegree', nargs='+', type=int, metavar='N', help='Fixed internal max-outdegree N, or inclusive MIN MAX')
    add('--large-nodes-per-subgraph', nargs=2, type=int, metavar=('MIN', 'MAX'), help='Size-shift range; default base MAX+1 through MAX+4')
    add('--density-outdegree', nargs='+', type=int, metavar='N', help='Density-shift fixed value/range; default base outdegree+1')
    add('--objectives', type=int, help='Default names water/time/energy, then risk and objective_N')
    add('--max-paths', type=int, help='Exact-enumeration path limit, enforced before materialization')
    add('--max-candidates', type=int, help='Rejection-sampling cap per split/domain')
    add('--difficulty-cutoff', type=int, help='Easy/hard boundary in exact source-target path count')
    for flag in ['--training-runs', '--training-seed', '--seed-stride', '--data-seed', '--suite-seed', '--eval-seed', '--bootstrap-seed']:
        add(flag, type=int)
    add('--training-seeds', type=csv_ints, help='Explicit independent training seeds; overrides generated seed sequence')
    add(curriculum_flag, key='curriculum', type=csv_ints, help='Requested-path k stages, e.g. 1,2,3; a single value fixes k')
    add('--eval-ks', type=csv_ints, help='Evaluation k values; defaults to distinct curriculum values')
    # Normalize these mutually exclusive budget controls after merging saved settings.
    budget = p.add_mutually_exclusive_group()
    budget.add_argument('--attempts-per-graph', '--traversals-per-graph', type=int, default=argparse.SUPPRESS,
                        help='Fixed B attempts per complete training task; must be >= every curriculum k')
    budget.add_argument('--attempt-multiplier', type=int, default=argparse.SUPPRESS,
                        help='Training B = multiplier * k [default: 8]')
    for flag in ['--steps', '--group-size', '--epochs-per-group', '--lora-rank', '--context-limit',
                 '--max-action-tokens', '--max-task-tokens', '--eval-repetitions', '--eval-relabelings', '--bootstrap-samples']:
        add(flag, type=int)
    for flag in ['--learning-rate', '--beta', '--clip', '--max-grad-norm', '--temperature']:
        add(flag, type=float)
    add('--pareto-solver', choices=['dijkstra', 'exhaustive', 'paretoset'],
        help='Solver for ground-truth Pareto paths in training, validation and evaluation graphs')
    add('--objective', choices=['pareto', 'hypervolume'], help='Training terminal reward; comparisons always report Pareto S')
    add('--eval-budget-multipliers', type=csv_ints, help='Evaluation B/k values; must include primary 8')
    return p


def resolve_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def resolve(args, explicit):
    # Relative CLI paths are anchored to the repository, regardless of cwd.
    output = resolve_path(args.output) if args.output is not None else None
    registry = {}
    if output is not None and (output/'run_manifest.json').exists():
        registry = json.loads((output/'run_manifest.json').read_text()).get('models', {})
    selected_model = explicit.get('model')
    if selected_model is None:
        if len(registry) > 1:
            raise ValueError('This run contains multiple models; select one with --model')
        selected_model = next(iter(registry), DEFAULTS['model'])
    # Settings are per model. Adding a second model must not overwrite the first.
    trained = output/f'{model_slug(selected_model)}_trained' if output is not None else None
    saved_path = trained/'launch_config.json' if trained is not None else None
    saved = json.loads(saved_path.read_text()) if saved_path is not None and saved_path.exists() else None
    settings = dict(DEFAULTS if saved is None else saved)
    if saved is not None:
        # Preserve replacement sampling when reopening historical runs.
        sampling={key:saved.get(key,False) for key in SAMPLING_DEFAULTS}
        saved.update(sampling);settings.update(sampling)
        saved = dict(saved, **thinking_options(saved))
        settings.update(thinking_options(saved))
        saved = dict(saved, **training_options(saved))
        settings.update(training_options(saved))
    if saved is None and registry:
        # Inherit the shared task design when adding a model to an existing run.
        anchor = json.loads(Path(next(iter(registry.values()))['launch_config']).read_text())
        shared = ('train_graphs','validation_graphs','eval_graphs','subgraphs','nodes_per_subgraph',
            'outdegree','large_nodes_per_subgraph','density_outdegree','objectives','max_paths',
            'max_candidates','difficulty_cutoff','data_seed','suite_seed','eval_seed','bootstrap_seed',
            'curriculum','eval_ks','eval_repetitions','eval_relabelings','eval_budget_multipliers',
            'bootstrap_samples','context_limit','history_format','temperature','max_action_tokens','max_task_tokens')
        settings.update({key:anchor.get(key,'full') if key=='history_format' else anchor[key] for key in shared})
        settings.update(thinking_options(anchor))
    settings['model'] = selected_model
    if any(model != selected_model and model_slug(model) == model_slug(selected_model) for model in registry):
        raise ValueError('Model basenames collide in this run; use distinct checkpoint basenames')
    settings.update({key: value for key, value in explicit.items() if key in DEFAULTS})
    if args.mode == 'eval' and saved is None:
        raise ValueError('--mode eval requires an existing model-specific launch_config.json')
    if 'model' in explicit and settings['model'] != (saved or DEFAULTS)['model'] and 'revision' not in explicit:
        raise ValueError('Changing --model also requires --revision for that checkpoint')
    if 'attempts_per_graph' in explicit:
        settings['attempt_multiplier'] = None
    elif 'attempt_multiplier' in explicit:
        settings['attempts_per_graph'] = None
    for key in ('subgraphs', 'outdegree', 'density_outdegree'):
        if settings[key] is not None and len(settings[key]) == 1:
            settings[key] = settings[key] * 2
    if settings['large_nodes_per_subgraph'] is None:
        upper = settings['nodes_per_subgraph'][1]
        settings['large_nodes_per_subgraph'] = [upper + 1, upper + 4]
    if settings['density_outdegree'] is None:
        settings['density_outdegree'] = [v + 1 for v in settings['outdegree']]
    if settings['eval_ks'] is None:
        settings['eval_ks'] = list(dict.fromkeys(settings['curriculum']))
    if settings['training_seeds'] is None:
        settings['training_seeds'] = [settings['training_seed'] + i * settings['seed_stride'] for i in range(settings['training_runs'])]
    elif 'training_seeds' in explicit and 'training_runs' not in explicit:
        settings['training_runs'] = len(settings['training_seeds'])
    if output is None:
        output = new_run_directory(resolve_path(args.runs_root), settings['model'])
    validate_run_directory(output, settings['model'])
    validate(settings, needs_evaluation=args.mode in ('all', 'eval', 'prepare'))
    if saved is not None and settings != saved:
        raise ValueError('Settings differ from this immutable run; choose a fresh --output directory')
    return output, settings, saved is not None


def validate(c, needs_evaluation=True):
    sampling_options(c)
    if c.get('pareto_solver', 'exhaustive') not in ('dijkstra', 'exhaustive', 'paretoset'):
        raise ValueError('Invalid pareto_solver')
    for key in ('train_graphs', 'validation_graphs', 'eval_graphs', 'objectives', 'max_paths', 'max_candidates',
                'training_runs', 'steps', 'group_size', 'epochs_per_group', 'lora_rank', 'context_limit',
                'max_action_tokens', 'max_task_tokens', 'eval_repetitions', 'eval_relabelings', 'bootstrap_samples'):
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    loading_contract(c)
    thinking_options(c)
    training_options(c)
    if c.get('history_format','full') not in ('full','delta','state_actions'):
        raise ValueError('history_format must be full, delta, or state_actions')
    if c['lora_last_layers'] < 0 or c['lora_alpha'] < 1 or c['min_free_gpu_gib'] < 0 or c['max_gpu_memory_gib'] < 0:
        raise ValueError('Invalid adapter layer/alpha or GPU memory controls')
    if c['group_size'] < 2:
        raise ValueError('GRPO group-size must be >= 2 to provide a group-relative learning signal')
    for key in ('data_seed', 'suite_seed', 'eval_seed', 'bootstrap_seed'):
        if not 0 <= c[key] < 2**63:
            raise ValueError(f'{key} must be in [0, 2**63)')
    for key in ('learning_rate', 'beta', 'clip', 'max_grad_norm', 'temperature'):
        if not math.isfinite(c[key]):
            raise ValueError(f'{key} must be finite')
    for key in ('subgraphs', 'nodes_per_subgraph', 'outdegree', 'large_nodes_per_subgraph', 'density_outdegree'):
        bounds = c[key]
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f'{key} requires inclusive MIN <= MAX')
    if c['subgraphs'][0] < 2:
        raise ValueError('At least two subgraphs are needed for this benchmark\'s reachable non-target sinks')
    if c['outdegree'][0] < 2 or c['outdegree'][1] >= c['nodes_per_subgraph'][0]:
        raise ValueError('Require 2 <= outdegree MIN <= MAX < nodes-per-subgraph MIN')
    if needs_evaluation:
        if c['eval_graphs'] % 10:
            raise ValueError('eval-graphs must be a multiple of 10 (five domains x two difficulties)')
        if c['density_outdegree'][0] < 2 or c['density_outdegree'][1] >= c['nodes_per_subgraph'][0]:
            raise ValueError('Density-shift outdegree must be >=2 and below nodes-per-subgraph MIN')
        if c['density_outdegree'][1] <= c['outdegree'][1]:
            raise ValueError('Density-shift outdegree MAX must exceed training outdegree MAX')
        if c['large_nodes_per_subgraph'][0] <= c['nodes_per_subgraph'][1]:
            raise ValueError('Size-shift node MIN must exceed training node MAX')
        if c['difficulty_cutoff'] < 1 or c['max_paths'] <= c['difficulty_cutoff']:
            raise ValueError('Require 1 <= difficulty-cutoff < max-paths to populate easy and hard strata')
    for key in ('curriculum', 'eval_ks', 'eval_budget_multipliers'):
        if not c[key] or any(v < 1 for v in c[key]):
            raise ValueError(f'{key} must contain positive integers')
    if len(set(c['eval_ks'])) != len(c['eval_ks']) or len(set(c['eval_budget_multipliers'])) != len(c['eval_budget_multipliers']):
        raise ValueError('Evaluation k values and budget multipliers must be distinct')
    if 8 not in c['eval_budget_multipliers']:
        raise ValueError('eval-budget-multipliers must include the primary 8k comparison')
    if len(c['training_seeds']) != c['training_runs'] or len(set(c['training_seeds'])) != c['training_runs']:
        raise ValueError('training-seeds must contain one distinct seed per training run')
    if any(seed < 0 or seed >= 2**63 for seed in c['training_seeds']):
        raise ValueError('Training seeds must be in [0, 2**63)')
    if c['attempts_per_graph'] is not None and c['attempts_per_graph'] < max(c['curriculum']):
        raise ValueError('attempts-per-graph must be >= every curriculum k')
    if c['attempts_per_graph'] is None and c['attempt_multiplier'] < 1:
        raise ValueError('attempt-multiplier must be positive')
    if c['max_paths'] < max(c['curriculum'] + c['eval_ks']) + 1:
        raise ValueError('max-paths must allow the requested frontier plus at least one dominated route')
    if c['learning_rate'] <= 0 or c['temperature'] <= 0 or c['max_grad_norm'] <= 0 or c['beta'] < 0 or not 0 < c['clip'] < 1:
        raise ValueError('Require lr/temperature/max-grad-norm > 0, beta >= 0, and 0 < clip < 1')
    if len(c['revision']) != 40 or any(v not in '0123456789abcdef' for v in c['revision']):
        raise ValueError('revision must be a full immutable 40-character commit SHA')


def configurations(output, c):
    def location(*parts):
        return str(output.joinpath(*parts))
    trained = output/f'{model_slug(c["model"])}_trained'
    frozen = output/f'{model_slug(c["model"])}_not_trained'
    graph = {key: c[key] for key in ('subgraphs', 'nodes_per_subgraph', 'outdegree',
             'large_nodes_per_subgraph', 'density_outdegree', 'objectives', 'max_paths')}
    graph['pareto_solver'] = c.get('pareto_solver', 'exhaustive')
    initial = initial_checkpoint(c)
    backend_fields = {key:c[key] for key in ('backend','compute_dtype','offload_embedding','lora_alpha','lora_last_layers','lora_mlp','min_free_gpu_gib','max_gpu_memory_gib')}
    backend_fields.update(thinking_options(c))
    backend_fields['history_format'] = c.get('history_format','full')
    train_configs, runs = [], []
    for seed in c['training_seeds']:
        run_dir = str(trained/'runs'/f'seed-{seed}')
        train = dict(model=c['model'],revision=c['revision'],**backend_fields, seed=seed, dataset=location('data', 'train.jsonl'), output=run_dir,
            initial_checkpoint=initial, curriculum=c['curriculum'], group_size=c['group_size'],
            steps=c['steps'], epochs_per_group=c['epochs_per_group'], learning_rate=c['learning_rate'],
            beta=c['beta'], clip=c['clip'], max_grad_norm=c['max_grad_norm'], lora_rank=c['lora_rank'],
            context_limit=c['context_limit'], temperature=c['temperature'], objective=c['objective'],
            pareto_solver=c.get('pareto_solver', 'exhaustive'),
            traversal_budget=c['attempts_per_graph'], attempt_multiplier=c['attempt_multiplier'])
        train.update(sampling_options(c))
        train.update(training_options(c),validation_dataset=location('data','validation.jsonl'),
            validation_ks=c['eval_ks'],max_action_tokens=c['max_action_tokens'],max_task_tokens=c['max_task_tokens'])
        train_configs.append((str(trained/'configs'/f'train_seed_{seed}.json'), train))
        runs.append(dict(id=f'grpo-{seed}', post_checkpoint=dict(initial, adapter=str(Path(run_dir)/'adapter')),
            training_config=str(Path(run_dir)/'config.json'), training_compute=str(Path(run_dir)/'training_compute.json'),
            selection=dict(split='validation', dataset=location('data', 'validation.jsonl'),
                criterion='maximum mean validation S at B=8k', evidence=str(Path(run_dir)/'validation_selection.json'))))
    template = json.loads((BASELINE_PRESET_DIR/'eval_baselines.json').read_text())
    template.update(**backend_fields,phase='confirmatory', graphs=c['eval_graphs'], repetitions=c['eval_repetitions'],
        relabelings=c['eval_relabelings'], training_run_count=c['training_runs'], training_runs=runs,
        initial_checkpoint=initial, pre_grpo_checkpoint=None, graph_generation=graph,
        ks=c['eval_ks'], budget_multipliers=c['eval_budget_multipliers'], difficulty_path_cutoff=c['difficulty_cutoff'],
        suite_seed=c['suite_seed'], evaluation_seed=c['eval_seed'], bootstrap_seed=c['bootstrap_seed'],
        bootstrap_samples=c['bootstrap_samples'], context_limit=c['context_limit'], temperature=c['temperature'],
        max_action_tokens=c['max_action_tokens'], max_task_tokens=c['max_task_tokens'],
        max_candidates_per_domain=c['max_candidates'], excluded_datasets=[location('data', 'train.jsonl')],
        validation_dataset=location('data', 'validation.jsonl'), pilot_suite=None,
        output=location('comparisons', model_slug(c['model'])), shared_suite=location('suite'),
        model_output_dirs={'initial':str(frozen/'evaluation'/'initial'),
                           **{r['id']:str(trained/'evaluation'/r['id']) for r in runs}})
    return graph, train_configs, template


def commands(output, train_configs, mode, no_plots):
    result = []
    if mode in ('all', 'train'):
        result.extend([sys.executable, '-m', MODULE+'.train', '--config', path] for path, _ in train_configs)
    if mode in ('all', 'eval'):
        config_dir = Path(train_configs[0][0]).parent
        original = str(config_dir/'eval.json')
        selected = str(config_dir/'eval.selected.json')
        base = [sys.executable, '-m', MODULE+'.eval_baselines']
        result.append(base+['select-validation', '--config', original, '--selected-config', selected])
        result.extend(base+[stage, '--config', selected] for stage in ('prepare', 'run', 'summarize'))
        if no_plots:
            result[-1].append('--no-plots')
    return result


def make_data(output, c, graph, exclusions=()):
    from .dataset import Instance, generate
    from .oracle import enumerate_paths
    from .graph_images import export_graph_image
    from .eval_baselines import IsomorphismIndex, digest, file_hash, jsonl, write_json
    directory = output/'data'
    directory.mkdir(exist_ok=True)
    manifest_path = directory/'manifest.json'
    spec = dict(graph_generation=graph, counts=dict(train=c['train_graphs'], validation=c['validation_graphs']),
                seed=c['data_seed'], k=max(c['curriculum']+c['eval_ks']))
    if exclusions:
        spec['prior_instances'] = sorted(digest(item.record()) for item in exclusions)
    sources = {name:file_hash(HERE/name) for name in ('dataset.py', 'generate_graph_world.py', 'mosp.py', 'oracle.py', 'graph_images.py')}
    sources['../generate_graph_world.py'] = file_hash(HERE.parent/'generate_graph_world.py')
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['spec'] != spec or saved.get('generation_sources') != sources or any(file_hash(directory/name) != value for name, value in saved['files'].items()):
            raise ValueError('Saved training/validation data changed; use a fresh output directory')
        return
    if any(directory.glob('*.jsonl')):
        raise ValueError('Incomplete existing dataset; use a fresh output directory')
    index, stats = IsomorphismIndex(), {}
    for item in exclusions:
        index.add_if_new(item)
    for offset, (split, count) in enumerate(spec['counts'].items()):
        rows, rejected, attempt = [], Counter(), 0
        while len(rows) < count:
            if attempt >= c['max_candidates']:
                raise RuntimeError(f'{split}: rejection cap exhausted: {dict(rejected)}')
            seed = c['data_seed']*2 + offset + attempt*2
            attempt += 1
            try:
                instance = generate(seed, k=spec['k'], graph_config=graph)
            except ValueError as error:
                rejected[str(error)] += 1
                continue
            if not index.add_if_new(instance):
                rejected['isomorphic_overlap'] += 1
                continue
            record = instance.record()
            image = directory/'images'/split/f'{digest(record)}.png'
            export_graph_image(instance, image)
            rows.append(record)
        jsonl(directory/f'{split}.jsonl', rows)
        stats[split] = dict(accepted=count, rejections=dict(rejected),
            instances=[dict(instance_id=digest(r), image=str(Path('images')/split/f'{digest(r)}.png'), seed=r['seed'], nodes=len(r['nodes']), edges=len(r['edges']),
                            paths=len(enumerate_paths(Instance.restore(r),c['max_paths']))) for r in rows])
    files = {f'{s}.jsonl': file_hash(directory/f'{s}.jsonl') for s in spec['counts']}
    files.update({str(image.relative_to(directory)): file_hash(image) for image in sorted((directory/'images').rglob('*.png'))})
    write_json(manifest_path, dict(spec=spec, statistics=stats, generation_sources=sources, files=files))


def dependencies(mode, no_plots, config=None):
    required = ['numpy', 'networkx', 'matplotlib', 'paretoset']
    if mode != 'prepare':
        required += ['torch', 'transformers', 'peft', 'accelerate']
        if loading_contract(config or DEFAULTS)['backend'] == 'unsloth_nf4':
            required += ['unsloth','bitsandbytes','fla','causal_conv1d']
        if mode in ('all', 'eval') and not no_plots:
            required.append('matplotlib')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(f'Missing packages: {", ".join(missing)}. Install {HERE / "requirements-eval.txt"} in this interpreter.')
    if mode != 'prepare':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Training/evaluation needs CUDA-enabled PyTorch and a visible GPU; prepare and dry-run work on CPU')


def verify_saved_suite(output, model):
    """Reuse must not silently associate old results with new weights or data."""
    from . import eval_baselines as evaluation
    comparison = output/'comparisons'/model_slug(model)
    selected = json.loads((output/f'{model_slug(model)}_trained'/'configs'/'eval.selected.json').read_text())
    manifest = json.loads((comparison/'suite_manifest.json').read_text())
    if evaluation.digest(selected) != manifest['config_id']:
        raise ValueError('Saved comparison controls changed; choose a fresh output directory')
    for name, expected in [('suite.jsonl', manifest['suite_id']), ('transcripts.jsonl',manifest['transcripts_id'])]:
        if evaluation.file_hash(comparison/name) != expected:
            raise ValueError('Saved held-out data changed; choose a fresh output directory')
    for name, expected in manifest.get('graph_images', {}).items():
        if evaluation.file_hash(comparison/name) != expected:
            raise ValueError('Saved graph images changed after suite creation')
    for name, expected in manifest['protocol_sources'].items():
        if evaluation.file_hash(HERE/name) != expected:
            raise ValueError('Protocol code changed after suite creation; choose a fresh output directory')
    for path, expected in manifest['excluded_files'].items():
        if evaluation.file_hash(path) != expected:
            raise ValueError('Training/validation data changed after suite creation')
    if (comparison/'completed.json').exists():
        previous = json.loads((comparison/'evaluation_manifest.json').read_text())
        current = {name:evaluation.checkpoint_id(spec) for name,spec in evaluation.checkpoint_specs(selected).items()}
        if current != previous['checkpoints']:
            raise ValueError('Checkpoint artifacts changed after evaluation; choose a fresh output directory')


def launch(args, explicit):
    timing = PipelineTiming(os.environ.get('MOSP_PIPELINE_STARTED_AT',PROCESS_STARTED_AT))
    with timing.component('Resolve settings and construct plan',announce=False):
        output, c, existing = resolve(args, explicit)
        graph, train_configs, evaluation = configurations(output, c)
        plan = dict(mode=args.mode, output=str(output), settings=c, graph_generation=graph,
                    training_configs={p: v for p, v in train_configs}, evaluation_config=evaluation,
                    commands=commands(output,train_configs,args.mode,args.no_plots),
                    guarantees='Frozen and GRPO models share one immutable held-out graph suite, k values and budgets; each explores independently.')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    status = 'failed'
    with artifact_environment(output, output/f'{model_slug(c["model"])}_trained'/'logs') as environment:
        try:
            _launch_actual(args, output, c, existing, train_configs, graph, evaluation, plan, environment, timing)
            status = 'completed'
        except KeyboardInterrupt:
            status = 'interrupted'
            print('run_mosp.sh: interrupted',file=sys.stderr,flush=True)
            raise SystemExit(130) from None
        except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as error:
            print(f'run_mosp.sh: {error}',file=sys.stderr,flush=True)
            raise SystemExit(2) from None
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1) from None
        finally:
            timing.finish(output,c['model'],args.mode,status)


def _launch_actual(args, output, c, existing, train_configs, graph, evaluation, plan, environment, timing):
    print(f'Run directory: {output}\nComplete console log: {output / "log.txt"}', flush=True)
    with timing.component('Dependency checks'):
        dependencies(args.mode, args.no_plots, c)
    # Shared data is validated before registering a new model or writing its
    # immutable settings, so an incompatible second model cannot pollute the run.
    with timing.component('Training/validation graph data and weighted images') as measurement:
        reused_data = (output/'data'/'manifest.json').exists()
        make_data(output,c,graph)
        if reused_data:
            measurement['status'] = 'reused'
    with timing.component('Write model registry and reproducible configurations'):
        trained = output/f'{model_slug(c["model"])}_trained'
        frozen = output/f'{model_slug(c["model"])}_not_trained'
        (trained/'configs').mkdir(parents=True, exist_ok=True)
        frozen.mkdir(exist_ok=True)
        manifest_path = output/'run_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
            run_directory=str(output), timestamp=output.name.removeprefix('run_'), timezone=RUN_TIMEZONE,
            cache_and_temp_paths=environment, models={})
        manifest['models'][c['model']] = dict(trained=str(trained), not_trained=str(frozen),
            launch_config=str(trained/'launch_config.json'))
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        if not existing:
            (trained/'launch_config.json').write_text(json.dumps(c,indent=2)+'\n')
        for path, config in train_configs:
            Path(path).write_text(json.dumps(config,indent=2)+'\n')
        (trained/'configs'/'eval.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    if args.mode == 'prepare':
        print(f'Prepared graph data and configurations in {output}')
        return
    training_progress = Progress(f'{c["model"]} | completed GRPO runs',len(train_configs)) if args.mode in ('all','train') else None
    def training_finished(config):
        training_progress.advance()
        if not args.no_plots:
            from .figures import training_curves
            with timed_component(f'Training metric figures | seed {config["seed"]}'):
                training_curves(config)
        if training_progress.completed==training_progress.total:
            training_model_report(c['model'],train_configs,trained)
    if args.mode=='eval':
        training_model_report(c['model'],train_configs,trained)
        if not args.no_plots:
            from .figures import training_curves
            with timing.component('Saved training metric figures'):
                for _,config in train_configs:training_curves(config)
    for command in plan['commands']:
        stage = 'train' if command[2] == MODULE+'.train' else command[3]
        label = {'train':f'GRPO training | {Path(command[-1]).stem}',
            'select-validation':'Validation checkpoint selection',
            'prepare':'Held-out suite, weighted images and external transcripts',
            'run':'Held-out evaluation | frozen and trained checkpoints',
            'summarize':'Comparison aggregation, confidence intervals and plots'}[stage]
        with timing.component(label) as measurement:
            if stage == 'train':
                # Reuse only a fully saved, configuration-matching training run.
                config = json.loads(Path(command[-1]).read_text())
                run_dir = Path(config['output'])
                if run_dir.exists():
                    if not (run_dir/'config.json').exists():
                        raise ValueError(f'Incomplete training run: {run_dir}; use a fresh output directory')
                    saved = json.loads((run_dir/'config.json').read_text())
                    if saved != config or not (run_dir/'training_state.pt').is_file() or not (run_dir/'adapter').is_dir() or (config.get('early_stopping') and not (run_dir/'training_summary.json').is_file()):
                        raise ValueError(f'Incomplete or changed training run: {run_dir}; use a fresh output directory')
                    measurement['status'] = 'reused'
                    print(f'Reusing completed training run: {run_dir}', flush=True)
                    summary = training_run_record(run_dir,config)
                    if summary:
                        task_tables(f'{c["model"]} | saved training seed {config["seed"]}',{'Run totals':summary},inference=False)
                    training_finished(config)
                    continue
            elif stage == 'prepare' and (Path(evaluation['output'])/'suite_manifest.json').exists():
                verify_saved_suite(output, c['model'])
                measurement['status'] = 'reused'
                print('Reusing the immutable held-out comparison suite',flush=True)
                continue
            elif stage == 'run' and (Path(evaluation['output'])/'completed.json').exists():
                verify_saved_suite(output, c['model'])
                measurement['status'] = 'reused'
                print('Reusing completed comparison results; summarize rechecks all task cells and hashes',flush=True)
                continue
            print('+ '+shlex.join(command), flush=True)
            # Each stage keeps a full console log under this same run directory.
            log_name = Path(command[-1]).stem if stage == 'train' else stage
            logdir = trained/'logs' if stage in ('train','select-validation') else Path(evaluation['output'])/'logs'
            logdir.mkdir(parents=True,exist_ok=True)
            execute(command, logdir/f'{log_name}.log')
            if stage=='train':
                training_finished(config)
    print(f'Finished {args.mode}: {output}', flush=True)


def execute(command, logfile):
    # Keep existing stage logs and also capture native child stdout/stderr in a
    # consistently named log.txt. A marker prevents duplicate child tee writes.
    logfile = Path(logfile)
    additional = logfile.parent/'log.txt'
    if len(command)>3 and command[1]=='-m' and command[2] in (MODULE+'.train',MODULE+'.eval_baselines',MODULE+'.eval_shaping'):
        config_path = Path(command[command.index('--config')+1])
        config = json.loads(config_path.read_text())
        additional = Path(config['output'])/'log.txt'
        if command[2]==MODULE+'.eval_baselines' and command[3]=='select-validation':
            selected_path = Path(command[command.index('--selected-config')+1])
            additional = selected_path.parent.parent/'validation'/'log.txt'
    with ExitStack() as stack:
        streams = []
        for path in dict.fromkeys(p.resolve() for p in (logfile,additional)):
            path.parent.mkdir(parents=True,exist_ok=True)
            streams.append(stack.enter_context(path.open('a',buffering=1)))
        environment = dict(os.environ,MOSP_CONSOLE_LOG_PATH=str(additional.resolve()))
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1,env=environment) as process:
            for line in process.stdout:
                print(line,end='',flush=True)
                for stream in streams:
                    stream.write(line)
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode,command)


def main():
    p = parser()
    args = p.parse_args()
    try:
        launch(args, vars(args))
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        p.exit(2,f'run_mosp.sh: {error}\n')


if __name__ == '__main__':
    main()
