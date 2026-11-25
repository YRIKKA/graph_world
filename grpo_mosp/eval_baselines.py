"""Paired, graph-clustered before/after evaluation of frozen Qwen checkpoints.

CLI stages: prepare -> run -> summarize; pilot-plan uses a separate pilot only.
Run as python -m grpo_mosp.eval_baselines --help.
No policy weights are downloaded during suite preparation or aggregation.
"""
import argparse
from .pipeline_timing import timed_component
from .model_options import add_thinking_arguments, apply_thinking_arguments, thinking_options
from contextlib import ExitStack
import os
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import networkx as nx
import numpy as np

from .dataset import Instance, generate, load
from .environment import Environment
from .memory import RULES
from .oracle import enumerate_paths, frontier
from .mosp import solve_pareto_paths
from .scoring import score
from .graph_images import export_graph_image
from .run_artifacts import BASELINE_PRESET_DIR, model_slug, console_log
from .reporting import Progress, Totals, checkpoint_report, comparison_report, model_name, table

HERE = Path(__file__).resolve().parent

DOMAINS = ('in_distribution', 'size', 'density', 'costs', 'objectives')
METRICS = ('S', 'complete_success', 'valid_path_fraction', 'duplicate_rate',
           'protocol_failure', 'hypervolume', 'normalized_hypervolume', 'A', 'L',
           'edge_traversals', 'generated_tokens', 'latency_seconds', 'limit_failure')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    # Stream checkpoint weights instead of reading multi-GB files into host RAM.
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify_protocol_sources(sources, error_message):
    """Check current source hashes and exact, documented relocation digests."""
    relocations = json.loads((HERE/'configs/metadata/source_relocations.json').read_text())
    for relocation in tuple(relocations.values()):
        relocations.setdefault(relocation['path'], relocation)
    for name, expected in sources.items():
        relocation = relocations.get(name)
        path = HERE/(relocation['path'] if relocation else name)
        actual = file_hash(path)
        if actual == expected:
            continue
        if relocation and actual == relocation['new_sha256'] and expected in relocation['previous_sha256']:
            continue
        raise ValueError(f'{error_message}: {name}')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def tagged_graph(instance):
    """Rooted, attributed representation invariant to presented node names."""
    graph = nx.DiGraph()
    for v in instance.graph:
        graph.add_node(v, role='source' if v == instance.source else 'target' if v == instance.target else 'other')
    for u, v, attrs in instance.graph.edges(data=True):
        graph.add_edge(u, v, costs=json.dumps([attrs.get(n, 1) for n in instance.names]))
    return graph


class IsomorphismIndex:
    """WL hashes are a prefilter, never the final exclusion test.

    Exact rooted weighted DAG isomorphism excludes relabeled copies. Objective
    names and order are part of the problem definition, but node IDs are not.
    """
    def __init__(self):
        self.buckets = defaultdict(list)

    def add_if_new(self, instance):
        graph = tagged_graph(instance)
        key = (instance.names, nx.weisfeiler_lehman_graph_hash(graph, node_attr='role', edge_attr='costs'))
        for prior in self.buckets[key]:
            if nx.is_isomorphic(graph, prior, node_match=nx.algorithms.isomorphism.categorical_node_match('role', ''),
                                 edge_match=nx.algorithms.isomorphism.categorical_edge_match('costs', '')):
                return False
        self.buckets[key].append(graph)
        return True


def excluded_paths(config):
    paths = set(config['excluded_datasets'])
    paths.add(config['validation_dataset'])
    if config.get('pilot_suite'):
        paths.add(config['pilot_suite'])
    for run in config['training_runs']:
        training_config = Path(run['training_config'])
        if training_config.exists():
            paths.add(json.loads(training_config.read_text())['dataset'])
    return sorted(paths)


def make_candidate(seed, domain, k, graph_config=None):
    spec = deepcopy(graph_config)
    if spec is not None and domain == 'density':
        spec['outdegree'] = spec.get('density_outdegree', [3, 3])
    item = generate(seed, k=k, larger=domain == 'size', extra_dimension=domain == 'objectives',
                    outdegree=3 if domain == 'density' else 2, graph_config=spec)
    if domain == 'costs':
        # This explicitly identified shift changes both range and relative totals
        # across different hop counts. Recompute global truth after transformation.
        for _, _, attrs in item.graph.edges(data=True):
            for name in item.names:
                attrs[name] = 2 * attrs[name] + 1
        item.max_cost = 2 * item.max_cost + 1
        paths = enumerate_paths(item, (spec or {}).get('max_paths', 4096))
        # Cost shifts change dominance, so recompute with the same chosen solver.
        solved = solve_pareto_paths(item.graph, item.source, item.target, weight=item.names,
                                   paths=paths, pareto_solver=(spec or {}).get('pareto_solver', 'exhaustive'))
        item.pareto = {tuple(path) for path, _ in solved}
        if len(item.pareto) < k or len(item.pareto) == len(paths):
            raise ValueError('shifted_frontier_rejected')
    return item


def relabel(instance, seed):
    item = deepcopy(instance)
    nodes = list(item.graph)
    shuffled = list(range(len(nodes)))
    random.Random(seed).shuffle(shuffled)
    mapping = {v: f'v{i}' for v, i in zip(nodes, shuffled)}
    item.graph = nx.relabel_nodes(item.graph, mapping)
    item.source, item.target = mapping[item.source], mapping[item.target]
    item.pareto = {tuple(mapping[v] for v in path) for path in item.pareto}
    return item


def _prepare(config, output):
    """Freeze suite and external explorer transcripts before any model evaluation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'suite.jsonl').exists():
        raise FileExistsError('Suite is immutable; use a fresh output directory')
    count = config['graphs']
    if count % (len(DOMAINS) * 2) or count < 10:
        raise ValueError('graphs must be a positive multiple of 10: five domains x two difficulties')
    exclusions = excluded_paths(config)
    index = IsomorphismIndex()
    for path in exclusions:
        for record in read_jsonl(path):
            index.add_if_new(Instance.restore(record.get('instance', record)))
    quota = count // (len(DOMAINS) * 2)
    accepted, rejection, rows = Counter(), Counter(), []
    candidate = config['suite_seed']
    with timed_component('Held-out graphs, exact frontiers and weighted images'):
        for domain in DOMAINS:
            attempts = 0
            while any(accepted[domain, d] < quota for d in ('easy', 'hard')):
                seed = candidate
                candidate += 1
                attempts += 1
                if attempts > config['max_candidates_per_domain']:
                    raise RuntimeError(f'Generation exhausted for {domain}: {dict(rejection)}')
                try:
                    item = make_candidate(seed, domain, max(config['ks']), config.get('graph_generation'))
                except ValueError as error:
                    rejection[domain + ':' + str(error)] += 1
                    continue
                paths = enumerate_paths(item, config.get('graph_generation', {}).get('max_paths', 4096))
                difficulty = 'easy' if len(paths) <= config['difficulty_path_cutoff'] else 'hard'
                if accepted[domain, difficulty] >= quota:
                    rejection[domain + ':stratum_full'] += 1
                    continue
                if not index.add_if_new(item):
                    rejection[domain + ':isomorphic_overlap'] += 1
                    continue
                family = digest(item.record())
                accepted[domain, difficulty] += 1
                for variant in range(config['relabelings']):
                    presented = item if variant == 0 else relabel(item, seed + variant * 10000019)
                    record = presented.record()
                    graph_id = digest(record)
                    # Export every held-out presentation, including relabeled copies
                    # and shifted edge costs, using the exact saved evaluator graph.
                    image = Path('images')/domain/f'{graph_id}.png'
                    export_graph_image(presented, output/image)
                    rows.append(dict(image=str(image), graph_id=graph_id, family_id=family, variant=variant,
                        domain=domain, difficulty=difficulty, paths=len(paths),
                        density=nx.density(item.graph), instance=record))
    # External exploration is fixed, observation-limited, uniform random, and
    # collected once. No outcome score ever selects a transcript or repetition.
    transcripts = []
    with timed_component('External random-exploration transcripts'):
        for row in rows:
            item = Instance.restore(row['instance'])
            for k in config['ks']:
                for multiplier in config['budget_multipliers']:
                    for repetition in range(config['repetitions']):
                        key = [row['graph_id'], k, multiplier, repetition]
                        seed = int(digest([config['evaluation_seed'], 'external', key])[:15], 16)
                        rng = random.Random(seed)
                        env = Environment(item, k, k * multiplier)
                        messages, moves = [dict(role='system', content=RULES)], []
                        while env.phase == 'traversal':
                            messages.append(dict(role='user', content=env.observation()))
                            action = rng.choice(env.choices())
                            moves.append(action)
                            messages.append(dict(role='assistant', content=json.dumps(action, separators=(',', ':'))))
                            env.step(action)
                        messages.append(dict(role='user', content=env.observation()))
                        transcripts.append(dict(key=key, seed=seed, moves=moves, history_sha256=digest(messages),
                            final_observation_sha256=digest(env.observation()), archive=[dict(path=list(p), cost=c) for p, c in sorted(env.archive.items())]))
    jsonl(output / 'suite.jsonl', rows)
    jsonl(output / 'transcripts.jsonl', transcripts)
    manifest = dict(phase=config['phase'], config=config, config_id=digest(config),
        suite_id=file_hash(output / 'suite.jsonl'), transcripts_id=file_hash(output / 'transcripts.jsonl'),
        excluded_files={p: file_hash(p) for p in exclusions}, rejections=dict(rejection),
        graphs=count, graph_presentations=len(rows),
        weighting='Within domain: difficulty weights then uniform k, equal graph families; repetitions and relabelings averaged within family',
        primary_budget_multiplier=8,
        graph_images={str(image.relative_to(output)):file_hash(image) for image in sorted((output/'images').rglob('*.png'))},
        protocol_sources={name: file_hash(Path(__file__).parent/name) for name in
            ('eval_baselines.py', 'policy.py', 'environment.py', 'memory.py', 'dataset.py',
             'generate_graph_world.py', 'oracle.py', 'scoring.py', 'graph_images.py', 'model_backends.py', 'model_defaults.py', 'history.py', 'reporting.py', 'model_options.py', 'pipeline_timing.py', '../generate_graph_world.py')})
    write_json(output / 'suite_manifest.json', manifest)
    return manifest


def suite_design_id(config):
    # Model identity and training hyperparameters do not define the graph suite.
    # Every property affecting graphs, repetitions or external transcripts does.
    keys = ('phase','graphs','repetitions','relabelings','ks','budget_multipliers',
        'difficulty_path_cutoff','difficulty_weights','suite_seed','evaluation_seed',
        'max_candidates_per_domain','graph_generation')
    return digest(dict(design={key:config.get(key) for key in keys},
        exclusions={path:file_hash(path) for path in excluded_paths(config)}))


def link_suite_assets(shared, output):
    output.mkdir(parents=True,exist_ok=True)
    for name in ('suite.jsonl','transcripts.jsonl','images'):
        destination = output/name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Suite is immutable; use a fresh comparison directory')
        destination.symlink_to(os.path.relpath(shared/name, output), target_is_directory=name=='images')


def prepare(config, output):
    """Share one frozen suite across every model in a timestamped run."""
    if not config.get('shared_suite'):
        return _prepare(config, output)
    shared, output = Path(config['shared_suite']), Path(output)
    design = suite_design_id(config)
    master = shared/'suite_manifest.json'
    if shared.exists():
        if not master.exists():
            raise ValueError('Shared suite is incomplete or another process is preparing it')
        manifest = json.loads(master.read_text())
        if manifest['suite_design_id'] != design:
            raise ValueError('Models in one run must use the exact same held-out data design')
        verify_protocol_sources(manifest['protocol_sources'], 'Shared suite protocol changed')
        for name, expected in [('suite.jsonl',manifest['suite_id']),('transcripts.jsonl',manifest['transcripts_id'])]:
            if file_hash(shared/name) != expected:
                raise ValueError('Shared suite data changed')
        for name, expected in manifest['graph_images'].items():
            if file_hash(shared/name) != expected:
                raise ValueError('Shared graph image changed')
        if (output/'suite_manifest.json').exists():
            raise FileExistsError('Suite is immutable; use a fresh comparison directory')
        link_suite_assets(shared, output)
        manifest = dict(manifest,config=config,config_id=digest(config))
        write_json(output/'suite_manifest.json',manifest)
        return manifest
    shared.parent.mkdir(parents=True,exist_ok=True)
    shared.mkdir()  # Exclusive reservation prevents concurrent different suites.
    try:
        manifest = _prepare(config, output)
        manifest.update(suite_design_id=design, shared_suite=str(shared))
        for name in ('suite.jsonl','transcripts.jsonl','images'):
            shutil.move(str(output/name), str(shared/name))
        link_suite_assets(shared, output)
        write_json(master, manifest)
        write_json(output/'suite_manifest.json',manifest)
        return manifest
    except Exception:
        # Remove only an empty reservation; keep any evidence of partial writes.
        if not any(shared.iterdir()):
            shared.rmdir()
        raise


def model_result_directory(config, output, model_id):
    if model_id in config.get('model_output_dirs', {}):
        return Path(config['model_output_dirs'][model_id])
    spec = checkpoint_specs(config)[model_id]
    trained = model_id != 'initial'
    return Path(output).parent/f'{model_slug(spec["model"])}_{"trained" if trained else "not_trained"}'/'evaluation'/model_id


def checkpoint_id(spec):
    """Immutable HF commits plus content hashes for every local model artifact."""
    result = deepcopy(spec)
    for field in ('model', 'adapter'):
        location = spec.get(field)
        if not location:
            continue
        if Path(location).is_dir():
            hashes = {str(f.relative_to(location)): file_hash(f) for f in sorted(Path(location).rglob('*')) if f.is_file()}
            if not hashes:
                raise ValueError(f'Empty checkpoint: {location}')
            result[field + '_files'] = hashes
        else:
            revision = spec.get('revision' if field == 'model' else 'adapter_revision', '')
            if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                raise ValueError(f'{field} must use an immutable 40-character revision')
    # Content identity, independent of local directory names: copying one model
    # to five paths must never masquerade as five independent training runs.
    identity = {}
    for field in ('model', 'adapter'):
        if spec.get(field):
            identity[field] = ({'files': result[field + '_files']} if field + '_files' in result
                else {'id': spec[field], 'revision': spec.get('revision' if field == 'model' else 'adapter_revision')})
    if 'loading' in spec:
        identity['loading'] = spec['loading']
    result['checkpoint_id'] = digest(identity)
    return result


def checkpoint_specs(config):
    specs = {'initial': config['initial_checkpoint']}
    if config.get('pre_grpo_checkpoint'):
        specs['pre_grpo'] = config['pre_grpo_checkpoint']
    specs.update({r['id']: r['post_checkpoint'] for r in config['training_runs']})
    if len(specs) != len(config['training_runs']) + 1 + bool(config.get('pre_grpo_checkpoint')):
        raise ValueError('Checkpoint IDs must be unique and cannot shadow initial/pre_grpo')
    return specs


def validate_runs(config):
    expected = config.get('pre_grpo_checkpoint') or config['initial_checkpoint']
    seen_seeds, selected = set(), []
    for run in config['training_runs']:
        training = json.loads(Path(run['training_config']).read_text())
        # Existing training runs initialize directly from their pinned base. SFT
        # runs must declare the immediate initialization descriptor explicitly.
        initialization = training.get('initial_checkpoint')
        if initialization is None:
            initialization = dict(model=training['model'], revision=training['revision'])
        if checkpoint_id(initialization)['checkpoint_id'] != checkpoint_id(expected)['checkpoint_id']:
            raise ValueError(f"{run['id']}: training initialization does not match the comparison checkpoint")
        if training['seed'] in seen_seeds:
            raise ValueError('Training runs must have distinct training seeds')
        seen_seeds.add(training['seed'])
        selection = run['selection']
        if selection['split'] != 'validation' or selection['dataset'] != config['validation_dataset']:
            raise ValueError('Checkpoint selection must use the declared validation data')
        if not selection.get('criterion'):
            raise ValueError('Record the prespecified checkpoint-selection criterion')
        evidence = Path(selection['evidence'])
        if not evidence.is_file():
            raise ValueError('Provide a validation selection record, not test-selected checkpoints')
        evidence_record = json.loads(evidence.read_text())
        if evidence_record['selected_checkpoint_id'] != checkpoint_id(run['post_checkpoint'])['checkpoint_id']:
            raise ValueError('Validation record selected a different checkpoint')
        if evidence_record['dataset_sha256'] != file_hash(config['validation_dataset']):
            raise ValueError('Validation data changed after checkpoint selection')
        base_contract = expected.get('loading')
        if base_contract != run['post_checkpoint'].get('loading'):
            raise ValueError('Frozen and post-trained models must use the exact same NF4/loading contract')
        compute_path = run.get('training_compute')
        compute = json.loads(Path(compute_path).read_text()) if compute_path else None
        selected.append(dict(id=run['id'], training_seed=training['seed'],
            training_config=training, selection=selection, selection_evidence_sha256=file_hash(evidence),
            training_compute=compute))
    if len(selected) != config['training_run_count']:
        raise ValueError('training_run_count does not match supplied independent checkpoints')
    return selected


def load_frozen(spec, config):
    """Both arms use the same NF4 base/config; only one has learned adapters."""
    from .model_backends import load_model
    from .policy import ConstrainedPolicy
    model,tokenizer = load_model(config,spec,training=False)
    return ConstrainedPolicy(model,tokenizer,config['context_limit'],config['temperature'],
        enable_thinking=getattr(model,'_mosp_loading',{}).get('enable_thinking'),
        history_format=config.get('history_format','full'), **thinking_options(config))


class EvaluationLimit(RuntimeError):
    pass


def limited_policy(policy, config):
    """Enforce identical action/task token limits while preserving base constraints."""
    # Wrap sampling rather than truncating generated JSON or observed history.
    from .policy import ConstrainedPolicy
    class Limited(ConstrainedPolicy):
        def __init__(self):
            super().__init__(policy.model, policy.tokenizer, policy.context_limit, policy.temperature, policy.enable_thinking, policy.history_format,
                             policy.reasoning_effort, policy.preserve_thinking)
            self.generated = self.action_generated = 0
        def sample(self, context, candidates):
            try:
                decision = super().sample(context, candidates)
            except RuntimeError as error:
                if str(error).startswith('Context overflow:'):
                    raise EvaluationLimit('context_limit') from error
                raise
            return decision
        def check_token_budget(self):
            if self.generated >= config['max_task_tokens'] or self.action_generated >= config['max_action_tokens']:
                raise EvaluationLimit('token_limit')
        def record_token(self):
            self.generated += 1
            self.action_generated += 1
        def action(self, messages, env):
            self.action_generated = 0
            return super().action(messages, env)
    return Limited()


def task_metrics(env, status):
    result = score(env._instance, env.archive, env.submission, env.k)
    slots = env.submission.get('paths', []) if isinstance(env.submission, dict) else []
    nonempty = [tuple(p) for p in slots if isinstance(p, list) and p and all(isinstance(v, str) for v in p)]
    return dict(S=result['pareto_fraction'], complete_success=result['all_correct'],
        valid_path_fraction=result['valid_slot_fraction'],
        duplicate_rate=(len(nonempty)-len(set(nonempty)))/env.k,
        protocol_failure=float(env.failure), hypervolume=result['hypervolume'],
        normalized_hypervolume=result['normalized_hypervolume'], A=result['discovery_fraction'],
        L=result['selection_gap'], edge_traversals=env.moves,
        limit_failure=float(status != 'ok'))


def sync(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def evaluate_task(policy, instance, k, multiplier, seed, config, transcript=None):
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    env = Environment(instance, k, k * multiplier)
    messages = [dict(role='system', content=RULES)]
    if transcript is not None:
        # Reconstruct and verify the frozen external archive without supplying
        # evaluator-only graph facts or frontier labels to either selector.
        for action in transcript['moves']:
            messages.append(dict(role='user', content=env.observation()))
            messages.append(dict(role='assistant', content=json.dumps(action, separators=(',', ':'))))
            env.step(action)
        if env.phase != 'submission':
            raise ValueError('External transcript must end at submission')
        if digest(env.observation()) != transcript['final_observation_sha256']:
            raise ValueError('External transcript observation mismatch')
        actual = [dict(path=list(p), cost=c) for p, c in sorted(env.archive.items())]
        if actual != transcript['archive']:
            raise ValueError('External transcript archive mismatch')
        if digest(messages + [dict(role='user', content=env.observation())]) != transcript['history_sha256']:
            raise ValueError('External transcript history mismatch')
    bounded = limited_policy(policy, config)
    status = 'ok'
    sync(config['device'])
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            while env.phase != 'terminal':
                messages.append(dict(role='user', content=env.observation()))
                action, _ = bounded.action(messages, env)
                messages.append(dict(role='assistant', content=action))
                env.step(action)
    except EvaluationLimit as error:
        # Prespecified intention-to-evaluate rule: keep the task with S=0 and
        # a separate limit flag. Never drop harder model-specific failures.
        status = str(error)
        env.submission, env.reward, env.phase = None, 0.0, 'terminal'
    sync(config['device'])
    elapsed = time.perf_counter() - started
    metrics = task_metrics(env, status)
    # The selection-only diagnostic inherits a fixed archive. Its inference
    # traversals are zero; external collection cost is recorded separately.
    metrics['external_edge_traversals'] = env.moves if transcript is not None else 0
    if transcript is not None:
        metrics['edge_traversals'] = 0
    metrics.update(generated_tokens=bounded.generated, latency_seconds=elapsed)
    result = dict(status=status, reward=metrics['S'], metrics=metrics, submission=env.submission,
        archive=[dict(path=list(p), cost=c) for p, c in sorted(env.archive.items())], history_sha256=digest(messages))
    if config.get('save_histories', False):
        result['messages'] = messages
    return result


def run(config, output, loader=None):
    """Evaluate one checkpoint at a time, with durable per-task JSONL records."""
    import gc
    import torch
    output = Path(output)
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    if manifest['config_id'] != digest(config):
        raise ValueError('Evaluation config differs from frozen suite config')
    for name, expected in [('suite.jsonl', manifest['suite_id']), ('transcripts.jsonl', manifest['transcripts_id'])]:
        if file_hash(output / name) != expected:
            raise ValueError(f'{name} changed after suite freezing')
    for path in excluded_paths(config):
        if manifest['excluded_files'].get(path) != file_hash(path):
            raise ValueError('Training/validation exclusions changed; prepare a fresh suite')
    verify_protocol_sources(manifest['protocol_sources'], 'Evaluation protocol source changed after suite freezing')
    for name, expected in manifest.get('graph_images', {}).items():
        if file_hash(output/name) != expected:
            raise ValueError('Held-out graph image changed after suite freezing')
    selected = validate_runs(config)
    specs = checkpoint_specs(config)
    identities = {key: checkpoint_id(spec) for key, spec in specs.items()}
    # Reject accidental aliases of one trained checkpoint as independent runs.
    trained_ids = [identities[r['id']]['checkpoint_id'] for r in config['training_runs']]
    if len(set(trained_ids)) != len(trained_ids):
        raise ValueError('Independent runs cannot share the same checkpoint artifact')
    result_file = output / 'tasks.jsonl'
    if result_file.exists():
        raise FileExistsError('Existing task results are immutable; use a fresh experiment directory')
    rows = read_jsonl(output / 'suite.jsonl')
    transcripts = {tuple(t['key']): t for t in read_jsonl(output / 'transcripts.jsonl')}
    write_json(output / 'evaluation_manifest.json', dict(config=config, suite_id=manifest['suite_id'],
        checkpoints=identities, training=selected,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, networkx=nx.__version__,
                     device=config['device'], cuda=torch.version.cuda),
        inference_costs='per-task seconds/tokens/traversals; loading and warmup recorded separately',
        grpo_attribution_reference='pre_grpo' if config.get('pre_grpo_checkpoint') else 'initial',
        uncertainty='paired graph-cluster percentile bootstrap; crossed training-run x graph bootstrap for procedure'))
    loader = loader or load_frozen
    loading = []
    with ExitStack() as streams:
        stream = streams.enter_context(result_file.open('w'))
        checkpoint_progress = Progress(f'{model_name(config)} | evaluated checkpoints',len(specs))
        for model_id, spec in specs.items():
            directory = model_result_directory(config,output,model_id)
            directory.mkdir(parents=True,exist_ok=True)
            with console_log(directory/'log.txt'), timed_component(f'Evaluation checkpoint | {model_id}'):
                individual_file = directory/'tasks.jsonl'
                if individual_file.exists():
                    raise FileExistsError(f'Existing model results are immutable: {individual_file}')
                individual = streams.enter_context(individual_file.open('w'))
                write_json(directory/'evaluation_manifest.json',dict(model_id=model_id,
                    checkpoint=identities[model_id], suite_id=manifest['suite_id'],config_id=digest(config),
                    shared_suite=config.get('shared_suite'), history_format=config.get('history_format','full'), controls={**thinking_options(config), **{key:config[key] for key in
                        ('ks','budget_multipliers','repetitions','context_limit','temperature','max_action_tokens','max_task_tokens')}}))
                progress = Progress(f'{spec["model"]} | {model_id} | evaluation tasks',
                    len(rows)*len(config['ks'])*len(config['budget_multipliers'])*config['repetitions']*2)
                groups = dict(all=defaultdict(Totals),domains=defaultdict(Totals))
                with timed_component(f'Evaluation model loading | {model_id}'):
                    started = time.perf_counter()
                    policy = loader(spec, config)
                    policy.model.requires_grad_(False)
                    policy.model.eval()
                    # Persist the actually loaded runtime/template identity for each arm.
                    write_json(directory/'backend.json',getattr(policy.model,'_mosp_backend_info',{}))
                    sync(config['device'])
                    load_seconds = time.perf_counter() - started
                # Fixed warmup, excluded from task latency and never oracle-selected.
                with timed_component(f'Evaluation warmup | {model_id}'):
                    started = time.perf_counter()
                    with torch.inference_mode():
                        tokens = policy.prompt([dict(role='user', content='Ready.')])
                        policy.model(input_ids=torch.tensor([tokens], device=policy.device), use_cache=False, logits_to_keep=1)
                    sync(config['device'])
                    loading.append(dict(model_id=model_id, load_seconds=load_seconds,
                                        warmup_seconds=time.perf_counter()-started))
                mode_seconds = defaultdict(float)
                for row in rows:
                    instance = Instance.restore(row['instance'])
                    for k in config['ks']:
                        for multiplier in config['budget_multipliers']:
                            for repetition in range(config['repetitions']):
                                for mode in ('independent_exploration', 'shared_transcript_selection'):
                                    task_started = time.perf_counter()
                                    key = [row['graph_id'], k, multiplier, repetition]
                                    seed = int(digest([config['evaluation_seed'], model_id, mode, key])[:15], 16)
                                    task = evaluate_task(policy, instance, k, multiplier, seed, config,
                                        transcripts[tuple(key)] if mode == 'shared_transcript_selection' else None)
                                    task.update(model_id=model_id, checkpoint_id=identities[model_id]['checkpoint_id'],
                                        graph_id=row['graph_id'], family_id=row['family_id'], variant=row['variant'],
                                        domain=row['domain'], difficulty=row['difficulty'], k=k, B=k*multiplier,
                                        budget_multiplier=multiplier, repetition=repetition, sampling_seed=seed, mode=mode)
                                    line = json.dumps(task) + '\n'
                                    stream.write(line)
                                    stream.flush()
                                    individual.write(line)
                                    individual.flush()
                                    measured = dict(task['metrics'],reward=task['metrics']['S'])
                                    groups['all'][mode,multiplier].add(measured)
                                    groups['domains'][mode,multiplier,row['domain']].add(measured)
                                    progress.advance()
                                    mode_seconds[mode] += time.perf_counter()-task_started
                individual.close()
                checkpoint_report(config,model_id,groups)
                from .reporting import table as print_table
                print_table(f'Evaluation component timing | {model_id}',['Component','Seconds'],
                    [['Model loading',f'{load_seconds:.3f}'],['Warmup',f'{loading[-1]["warmup_seconds"]:.3f}'],
                     *[[mode,f'{seconds:.3f}'] for mode,seconds in mode_seconds.items()]])
                loading[-1]['evaluation_mode_wall_seconds'] = dict(mode_seconds)
                write_json(directory/'console_summary.json',dict(model_id=model_id,
                    by_mode_budget=[dict(mode=m,budget_multiplier=b,**v.record()) for (m,b),v in groups['all'].items()],
                    by_domain=[dict(mode=m,budget_multiplier=b,domain=d,**v.record()) for (m,b,d),v in groups['domains'].items()]))
                write_json(directory/'inference_loading.json',loading[-1])
                write_json(directory/'completed.json',dict(tasks_sha256=file_hash(individual_file),
                    task_count=len(rows)*len(config['ks'])*len(config['budget_multipliers'])*config['repetitions']*2))
                del policy
                gc.collect()
                if str(config['device']).startswith('cuda'):
                    torch.cuda.empty_cache()
                write_json(output / 'inference_loading.json', loading)
            checkpoint_progress.advance()
    # Partial/aborted runs have no completion marker and cannot be summarized.
    write_json(output / 'completed.json', dict(tasks_sha256=file_hash(result_file), task_count=len(rows)*len(specs)*len(config['ks'])*len(config['budget_multipliers'])*config['repetitions']*2))


def validate_results(rows, config, suite, model_ids):
    expected = {(m, g['graph_id'], k, b, q, mode)
        for m in model_ids for g in suite for k in config['ks']
        for b in config['budget_multipliers'] for q in range(config['repetitions'])
        for mode in ('independent_exploration', 'shared_transcript_selection')}
    seen = set()
    metadata = {g['graph_id']: g for g in suite}
    for row in rows:
        key = (row['model_id'], row['graph_id'], row['k'], row['budget_multiplier'], row['repetition'], row['mode'])
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected result cell')
        seen.add(key)
        graph = metadata[row['graph_id']]
        if any(row[field] != graph[field] for field in ('family_id', 'variant', 'domain', 'difficulty')):
            raise ValueError('Task graph metadata mismatch')
        if row['B'] != row['k'] * row['budget_multiplier']:
            raise ValueError('Traversal budget mismatch')
        if not all(np.isfinite(row['metrics'][m]) for m in METRICS):
            raise ValueError('Non-finite task metrics')
    if seen != expected:
        raise ValueError(f'Incomplete evaluation: missing {len(expected - seen)} task cells')


def family_means(rows, ks):
    """First repetitions, then relabelings, then prespecified uniform k weights.

    Never pool repetitions as independent graphs, or treat baseline repetitions
    reused across training runs as independent baseline checkpoint observations.
    """
    by_variant = defaultdict(list)
    for row in rows:
        key = row['model_id'], row['family_id'], row['k'], row['variant']
        by_variant[key].append([row['metrics'][m] for m in METRICS])
    by_k = defaultdict(list)
    for (model, family, k, _), values in by_variant.items():
        by_k[model, family, k].append(np.mean(values, axis=0))
    by_family = defaultdict(dict)
    for (model, family, k), values in by_k.items():
        by_family[model, family][k] = np.mean(values, axis=0)
    result = {}
    for key, values in by_family.items():
        if set(values) != set(ks):
            raise ValueError('Missing k stratum')
        result[key] = np.mean([values[k] for k in ks], axis=0)
    return result


def bootstrap_pair(means, metadata, posts, reference, weights, samples=2000, seed=0,
                   draw_runs=False, projected_graphs=None, projected_runs=None):
    """Paired stratified cluster bootstrap, crossed with independent training runs.

    A graph draw is shared by every model/run. A training-run draw is shared
    across every graph stratum. Entire repetitions and relabelings have already
    been averaged within family, so all their correlations remain clustered.
    """
    families = sorted(metadata)
    baseline = np.stack([means[reference, f] for f in families])
    after = np.stack([[means[post, f] for f in families] for post in posts])
    groups = {stratum: [i for i, f in enumerate(families)
                       if (metadata[f]['domain'], metadata[f]['difficulty']) == stratum]
              for stratum in weights}
    if any(not indices for indices in groups.values()):
        raise ValueError('Missing prespecified graph stratum')
    def estimate(run_ids, graph_groups):
        base = sum(weights[s] * baseline[indices].mean(axis=0) for s, indices in graph_groups.items())
        post = sum(weights[s] * after[run_ids][:, indices].mean(axis=(0, 1)) for s, indices in graph_groups.items())
        return np.stack([base, post, post - base])
    point = estimate(np.arange(len(posts)), groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        run_ids = rng.choice(len(posts), projected_runs or len(posts), replace=True) if draw_runs else np.arange(len(posts))
        graph_groups = {s: rng.choice(ids, max(1, round(projected_graphs * weights[s])) if projected_graphs else len(ids), replace=True)
                        for s, ids in groups.items()}
        draws.append(estimate(run_ids, graph_groups))
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return point, low, high


def summary_rows(rows, suite, config):
    result = []
    references = ['initial'] + (['pre_grpo'] if config.get('pre_grpo_checkpoint') else [])
    run_ids = [r['id'] for r in config['training_runs']]
    entities = [(r, [r], False) for r in run_ids] + [('training_procedure', run_ids, True)]
    scopes = [(d, [d]) for d in DOMAINS] + [('distribution_shift_macro', list(DOMAINS[1:]))]
    progress = Progress(f'{model_name(config)} | aggregation estimates',
        len(scopes)*3*(len(config['ks'])+1)*len(config['budget_multipliers'])*2*len(references)*len(entities))
    for scope, domains in scopes:
        for difficulty in ('all', 'easy', 'hard'):
            difficulties = ('easy', 'hard') if difficulty == 'all' else (difficulty,)
            normalizer = sum(config['difficulty_weights'][d] for d in difficulties)
            weights = {(domain, d): config['difficulty_weights'][d] / normalizer / len(domains)
                       for domain in domains for d in difficulties}
            selected_suite = [g for g in suite if (g['domain'], g['difficulty']) in weights]
            metadata = {g['family_id']: g for g in selected_suite}
            for k_group in ['all', *config['ks']]:
                ks = config['ks'] if k_group == 'all' else [k_group]
                for multiplier in config['budget_multipliers']:
                    for mode in ('independent_exploration', 'shared_transcript_selection'):
                        subset = [r for r in rows if r['family_id'] in metadata and r['k'] in ks
                                  and r['budget_multiplier'] == multiplier and r['mode'] == mode]
                        means = family_means(subset, ks)
                        for reference in references:
                            for entity, posts, crossed in entities:
                                key = [scope, difficulty, k_group, multiplier, mode, reference, entity]
                                point, low, high = bootstrap_pair(means, metadata, posts, reference, weights,
                                    config['bootstrap_samples'], int(digest([config['bootstrap_seed'], key])[:15], 16), crossed)
                                for index, metric in enumerate(METRICS):
                                    if scope == 'distribution_shift_macro' and metric == 'hypervolume':
                                        continue  # Raw volumes across different dimensions have incompatible units.
                                    result.append(dict(scope=scope, difficulty=difficulty, k=k_group,
                                        budget_multiplier=multiplier, primary=multiplier == 8,
                                        mode=mode, reference=reference, comparison=entity, metric=metric,
                                        baseline=float(point[0, index]), post=float(point[1, index]),
                                        difference=float(point[2, index]), ci_low=float(low[2, index]), ci_high=float(high[2, index]),
                                        baseline_ci_low=float(low[0, index]), baseline_ci_high=float(high[0, index]),
                                        post_ci_low=float(low[1, index]), post_ci_high=float(high[1, index]),
                                        graph_families=len(metadata), training_runs=len(posts),
                                        bootstrap='crossed_runs_graphs' if crossed else 'paired_graphs'))
                                progress.advance()
    return result


def plot_budget(table, output):
    # Standard plotting artifacts (SVG + PNG), suitable for exporting/sharing.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    references = sorted({r['reference'] for r in table})
    for reference in references:
        fig, axes = plt.subplots(len(DOMAINS), 2, figsize=(11, 3 * len(DOMAINS)), squeeze=False)
        for row_index, domain in enumerate(DOMAINS):
            selected = sorted([r for r in table if r['scope'] == domain and r['difficulty'] == 'all'
                and r['k'] == 'all' and r['mode'] == 'independent_exploration' and r['metric'] == 'S'
                and r['comparison'] == 'training_procedure' and r['reference'] == reference], key=lambda r: r['budget_multiplier'])
            x = [r['budget_multiplier'] for r in selected]
            left, right = axes[row_index]
            for prefix, label in [('baseline', reference), ('post', 'GRPO: independent-run mean')]:
                left.plot(x, [r[prefix] for r in selected], marker='o', label=label)
                left.fill_between(x, [r[prefix + '_ci_low'] for r in selected], [r[prefix + '_ci_high'] for r in selected], alpha=0.15)
            right.plot(x, [r['difference'] for r in selected], marker='o')
            right.fill_between(x, [r['ci_low'] for r in selected], [r['ci_high'] for r in selected], alpha=0.2)
            right.axhline(0, color='black', linewidth=0.8)
            left.set(title=domain, ylabel='Mean correct-slot fraction S', ylim=(-0.02, 1.02))
            right.set(title=domain, ylabel='Paired improvement in S')
            for ax in (left, right):
                ax.set_xscale('log', base=2)
                ax.set_xticks(x, [str(v) for v in x])
                ax.set_xlabel('Traversal budget B / k (primary: 8)')
                ax.axvline(8, color='gray', linestyle=':', linewidth=0.8)
            left.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(output) / f'performance_vs_budget_{reference}.svg')
        fig.savefig(Path(output) / f'performance_vs_budget_{reference}.png', dpi=160)
        plt.close(fig)


def checked_results(output, config):
    output = Path(output)
    completed = json.loads((output / 'completed.json').read_text())
    if file_hash(output / 'tasks.jsonl') != completed['tasks_sha256']:
        raise ValueError('Task results changed after evaluation completed')
    rows = read_jsonl(output / 'tasks.jsonl')
    suite = read_jsonl(output / 'suite.jsonl')
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    if manifest['config_id'] != digest(config) or file_hash(output / 'suite.jsonl') != manifest['suite_id']:
        raise ValueError('Suite/config does not match frozen experiment')
    if len(rows) != completed['task_count']:
        raise ValueError('Incorrect completed task count')
    validate_results(rows, config, suite, checkpoint_specs(config))
    return rows, suite


def summarize(config, output, plots=True):
    rows, suite = checked_results(output, config)
    with timed_component('Paired comparison aggregation and bootstrap confidence intervals'):
        table = summary_rows(rows, suite, config)
    output = Path(output)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_model_tables(table, config, output)
    comparison_report(config,rows,table)
    primary = [r for r in table if r['primary'] and r['difficulty'] == 'all' and r['k'] == 'all'
               and r['metric'] == 'S' and r['mode'] == 'independent_exploration']
    lines = ['# Frozen-checkpoint comparison', '',
        'Primary: B=8k. Uniform k weights, configured difficulty weights, equal graph-family weights within each stratum.',
        'Repetitions and relabelings are averaged within graph families. Intervals are pointwise 95% percentile bootstrap intervals.',
        'In-distribution and shifted domains are reported separately; shift macro averages the four shifted domains equally.', '',
        '| Domain | Comparison | Reference | Before S | After S | Paired difference [95% CI] |',
        '|---|---|---|---:|---:|---:|']
    for r in primary:
        lines.append(f"| {r['scope']} | {r['comparison']} | {r['reference']} | {r['baseline']:.4f} | {r['post']:.4f} | {r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    lines += ['', 'All secondary metrics, k/difficulty strata, budgets and the shared-transcript diagnostic are in comparison.csv.',
        'Training compute is in evaluation_manifest.json; model loading/warmup is in inference_loading.json; inference costs are per task.',
        'No checkpoint, repetition or budget is selected using test outcomes. Five runs remain a limited sample of the training procedure.',
        'A positive point estimate alone is not evidence that the procedure improves reliably; inspect the paired interval and domain-specific effects.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    if plots:
        from .figures import evaluation_bars,figure_root
        figure_directory=figure_root(config,output)/model_slug(config['initial_checkpoint']['model'])/'evaluation/budget_curves'
        figure_directory.mkdir(parents=True,exist_ok=True)
        with timed_component('Performance-versus-budget plot export'):
            plot_budget(table, figure_directory)
        with timed_component('All-checkpoint metric bar charts'):
            evaluation_bars(config,table,output)
    return table


def write_model_tables(table, config, output):
    """Put each checkpoint's own means/intervals beside its own task results."""
    for model_id in checkpoint_specs(config):
        reference = model_id if model_id in ('initial','pre_grpo') else 'initial'
        comparison = 'training_procedure' if model_id in ('initial','pre_grpo') else model_id
        prefix = 'baseline' if model_id in ('initial','pre_grpo') else 'post'
        selected = [r for r in table if r['reference']==reference and r['comparison']==comparison]
        rows = [{**{key:r[key] for key in ('scope','difficulty','k','budget_multiplier','mode','metric')},
                 'mean':r[prefix], 'ci_low':r[prefix+'_ci_low'],'ci_high':r[prefix+'_ci_high']}
                for r in selected]
        directory = model_result_directory(config,output,model_id)
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'metrics.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def select_validation(config, candidates, selected_config_path):
    """Choose among candidate checkpoints using ONLY fixed validation averages.

    The held-out test suite is neither loaded nor scored by this command.
    Candidate lists and the selection rule must be fixed before running it.
    """
    import gc
    import torch
    selected = deepcopy(config)
    instances = load(config['validation_dataset'])
    if not instances or any(len(item.pareto) < max(config['ks']) for item in instances):
        raise ValueError('Validation requires nonempty data with enough Pareto paths')
    # Validation must also be disjoint from training, including relabelings.
    index = IsomorphismIndex()
    training_paths = set(config['excluded_datasets'])
    candidate_configs = [run['training_config'] for run in config['training_runs']]
    candidate_configs.extend(option['training_config'] for options in candidates.values()
                             for option in options if 'checkpoint' in option and option.get('training_config'))
    for path in candidate_configs:
        training_paths.add(json.loads(Path(path).read_text())['dataset'])
    for path in sorted(training_paths):
        for record in read_jsonl(path):
            index.add_if_new(Instance.restore(record.get('instance', record)))
    for item in instances:
        if not index.add_if_new(item):
            raise ValueError('Validation overlaps training or contains duplicate graph copies')
    for run in selected['training_runs']:
        options = candidates.get(run['id'], [run['post_checkpoint']])
        evaluated = []
        progress = Progress(f'{model_name(config)} | {run["id"]} | validation tasks',
            len(options)*len(instances)*len(config['ks'])*config['repetitions'])
        for option in options:
            with timed_component(f'Validation candidate | {run["id"]}'):
                spec = option.get('checkpoint', option)
                identity = checkpoint_id(spec)
                policy = load_frozen(spec, config)
                values = []
                for item in instances:
                    for k in config['ks']:
                        for repetition in range(config['repetitions']):
                            seed = int(digest(['validation', config['evaluation_seed'], run['id'], item.seed, k, repetition])[:15], 16)
                            result = evaluate_task(policy, item, k, 8, seed, config)
                            values.append(dict(graph_id=digest(item.record()), k=k, repetition=repetition, S=result['metrics']['S']))
                            progress.advance()
                evaluated.append(dict(spec=spec, checkpoint_id=identity['checkpoint_id'],
                                      training_config=option.get('training_config', run['training_config']),
                                      training_compute=option.get('training_compute', run.get('training_compute')),
                                      mean_S=float(np.mean([v['S'] for v in values])), tasks=values))
                del policy
                gc.collect()
                if str(config['device']).startswith('cuda'):
                    torch.cuda.empty_cache()
        # Deterministic tie break by immutable identity, never test performance.
        winner = sorted(evaluated, key=lambda item: (-item['mean_S'], item['checkpoint_id']))[0]
        table(f'{model_name(config)} | {run["id"]} | validation only',
              ['Checkpoint','Tasks','Total reward S','Mean S','Selected'],
              [[v['checkpoint_id'][:16],len(v['tasks']),f"{sum(t['S'] for t in v['tasks']):.4f}",
                f"{v['mean_S']:.4f}",'yes' if v is winner else ''] for v in evaluated])
        run['post_checkpoint'] = winner['spec']
        run['training_config'] = winner['training_config']
        run['training_compute'] = winner['training_compute']
        evidence = Path(run['selection']['evidence'])
        evidence.parent.mkdir(parents=True, exist_ok=True)
        training_output = json.loads(Path(winner['training_config']).read_text()).get('output')
        stopping_path = Path(training_output)/'early_stopping.json' if training_output else None
        write_json(evidence, dict(dataset_sha256=file_hash(config['validation_dataset']),
            criterion='maximum mean validation S at B=8k, uniform graph/k/repetition weights; checkpoint-ID tie break',
            selected_checkpoint_id=winner['checkpoint_id'], candidates=evaluated,
            training_early_stopping=json.loads(stopping_path.read_text()) if stopping_path is not None and stopping_path.exists() else None))
        run['selection']['criterion'] = 'maximum mean validation S at B=8k'
    validate_runs(selected)
    write_json(selected_config_path, selected)
    return selected


def pilot_plan(config, output):
    """Empirical crossed-bootstrap precision projection from a SEPARATE pilot.

    This is an exploratory allocation aid, not a power guarantee. It does not
    modify the confirmatory configuration or select its checkpoints on test data.
    Repetitions remain fixed; proposed extra training runs must actually be trained.
    """
    if config['phase'] != 'pilot':
        raise ValueError('Sample-size planning must use a separate pilot experiment')
    rows, suite = checked_results(output, config)
    subset = [r for r in rows if r['domain'] == 'in_distribution' and r['budget_multiplier'] == 8
              and r['mode'] == 'independent_exploration']
    metadata = {g['family_id']: g for g in suite if g['domain'] == 'in_distribution'}
    means = family_means(subset, config['ks'])
    weights = {('in_distribution', d): w for d, w in config['difficulty_weights'].items()}
    posts = [r['id'] for r in config['training_runs']]
    plans = []
    for graphs in config['pilot_graph_grid']:
        for runs in config['pilot_run_grid']:
            point, low, high = bootstrap_pair(means, metadata, posts,
                'pre_grpo' if config.get('pre_grpo_checkpoint') else 'initial', weights,
                config['bootstrap_samples'], config['bootstrap_seed'], True, graphs, runs)
            halfwidth = float((high[2, 0] - low[2, 0]) / 2)
            plans.append(dict(in_distribution_graphs=graphs, independent_training_runs=runs,
                repetitions=config['repetitions'], projected_S_ci_halfwidth=halfwidth,
                meets_target=halfwidth <= config['pilot_target_halfwidth']))
    result = dict(pilot_suite_id=json.loads((Path(output) / 'suite_manifest.json').read_text())['suite_id'],
        target_halfwidth=config['pilot_target_halfwidth'], plans=plans,
        limitations='Empirical pilot resampling cannot discover unseen run variability; projections are exploratory, especially with few runs. Freeze a new confirmatory configuration before test evaluation; never reuse pilot graph families.')
    write_json(Path(output) / 'pilot_plan.json', result)
    return result


def validate_config(config):
    thinking_options(config)
    for key in ('graphs', 'repetitions', 'relabelings', 'training_run_count', 'bootstrap_samples', 'context_limit', 'max_task_tokens', 'max_action_tokens'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    budgets = config['budget_multipliers']
    if not budgets or len(set(budgets)) != len(budgets) or any(not isinstance(b, int) or b < 1 for b in budgets) or 8 not in budgets:
        raise ValueError('Budget multipliers must be distinct positive integers and include primary B=8k')
    if not config['ks'] or len(set(config['ks'])) != len(config['ks']) or any(k < 1 for k in config['ks']):
        raise ValueError('ks must be distinct positive integers')
    if config['phase'] not in ('pilot', 'confirmatory'):
        raise ValueError('phase must be pilot or confirmatory')
    if set(config['difficulty_weights']) != {'easy', 'hard'} or any(w <= 0 for w in config['difficulty_weights'].values()) or not np.isclose(sum(config['difficulty_weights'].values()), 1):
        raise ValueError('Difficulty weights must be positive and sum to one')
    if config['temperature'] <= 0 or config['training_run_count'] != len(config['training_runs']):
        raise ValueError('Invalid sampling temperature or training run count')
    checkpoint_specs(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select-validation', 'run', 'summarize', 'pilot-plan'])
    parser.add_argument('--config', default=BASELINE_PRESET_DIR/'eval_baselines.json')
    parser.add_argument('--output', help='Immutable experiment directory; defaults to config.output')
    parser.add_argument('--candidates', help='JSON mapping training-run IDs to candidate checkpoint descriptor lists')
    parser.add_argument('--selected-config', help='Output config after validation selection')
    parser.add_argument('--no-plots', action='store_true')
    add_thinking_arguments(parser)
    args = parser.parse_args()
    config = apply_thinking_arguments(json.loads(Path(args.config).read_text()), args)
    validate_config(config)
    output = args.output or config['output']
    log_directory = Path(args.selected_config).parent.parent/'validation' if args.stage=='select-validation' and args.selected_config else Path(output)
    with console_log(log_directory/'log.txt'):
        try:
            if args.stage == 'prepare':
                print(json.dumps(prepare(config, output), indent=2))
            elif args.stage == 'select-validation':
                if not args.selected_config:
                    parser.error('select-validation requires --selected-config')
                candidates = json.loads(Path(args.candidates).read_text()) if args.candidates else {}
                select_validation(config, candidates, args.selected_config)
            elif args.stage == 'run':
                run(config, output)
            elif args.stage == 'summarize':
                summarize(config, output, not args.no_plots)
            else:
                print(json.dumps(pilot_plan(config, output), indent=2))

        except KeyboardInterrupt:
            print('Evaluation interrupted',flush=True)
            raise SystemExit(130) from None
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(1) from None


if __name__ == '__main__':
    main()
