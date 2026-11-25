"""Graph stages and deterministic rehearsal; dependency-free until data loading.

Structural complexity increases with the supplied parameters. Realized graph
difficulty and Pareto-frontier size need not increase monotonically.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

KEYS=('subgraphs','nodes-per-subgraph','outdegree')
OVERRIDDEN_GRAPH_KEYS=('subgraphs','nodes_per_subgraph','outdegree','objectives',
                       'large_nodes_per_subgraph','density_outdegree')
REPLAY_PROBABILITY=.2


def validate_parameters(parameters):
    """Reject malformed stage lists before generation or GPU model loading."""
    if not isinstance(parameters,dict) or set(parameters)!=set(KEYS):
        raise ValueError(f'Graph curriculum must contain exactly {KEYS}; none may be omitted')
    values=[parameters[key] for key in KEYS]
    if any(not isinstance(value,list) or not value for value in values):
        raise ValueError('Every graph curriculum parameter must be a nonempty list')
    if len({len(value) for value in values})!=1:
        raise ValueError('Graph curriculum parameter lists must have the same length')
    stages=[]
    previous=None
    for index,(subgraphs,nodes,outdegree) in enumerate(zip(*values)):
        if type(subgraphs) is not int or subgraphs<2:
            raise ValueError(f'Stage {index+1}: subgraphs must be an integer >= 2')
        if not isinstance(nodes,list) or len(nodes)!=2 or any(type(n) is not int for n in nodes):
            raise ValueError(f'Stage {index+1}: nodes-per-subgraph must be [MIN, MAX] integers')
        if nodes[0]<3 or nodes[0]>nodes[1]:
            raise ValueError(f'Stage {index+1}: require 3 <= node MIN <= MAX')
        if type(outdegree) is not int or not 2<=outdegree<nodes[0]:
            raise ValueError(f'Stage {index+1}: require 2 <= outdegree < node MIN')
        dimensions=(subgraphs,*nodes,outdegree)
        if previous is not None and any(a<b for a,b in zip(dimensions,previous)):
            raise ValueError('Order graph stages from simple to complex: parameters must not decrease')
        previous=dimensions
        stages.append(dict(subgraphs=[subgraphs,subgraphs],nodes_per_subgraph=list(nodes),outdegree=[outdegree,outdegree]))
    return stages


def read_config(filename):
    path=Path(filename).expanduser().resolve()
    try:
        raw=path.read_bytes()
        parameters=json.loads(raw)
    except (OSError,ValueError) as error:
        raise ValueError(f'Cannot read graph curriculum {path}: {error}') from error
    return dict(source_file=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),
                parameters=parameters,stages=validate_parameters(parameters),replay_probability=REPLAY_PROBABILITY)


def graph_spec(stage, settings):
    result=dict(stage,objectives=settings['objectives'],max_paths=settings['max_paths'],
                pareto_solver=settings.get('pareto_solver', 'exhaustive'))
    result['large_nodes_per_subgraph']=[stage['nodes_per_subgraph'][1]+1,stage['nodes_per_subgraph'][1]+4]
    result['density_outdegree']=[stage['outdegree'][0]+1,stage['outdegree'][1]+1]
    return result


def stage_datasets(output, settings):
    stages=settings['graph_curriculum']['stages']
    total=settings['train_graphs']
    # Keep --train-graphs as the TOTAL count, split evenly across graph stages.
    return [dict(stage=i,dataset=str(Path(output)/'data/graph_stages'/f'stage-{i}'/'data/train.jsonl'),
                 count=total//len(stages)+int(i<total%len(stages)),graph_generation=graph_spec(stage,settings))
            for i,stage in enumerate(stages)]


def make_data(output, settings):
    """Reuse original generation/images with exclusions spanning all stages.

    data/train.jsonl is the union for provenance and held-out exclusions. Fixed
    validation graphs follow the final stage, never the changing training stage.
    """
    from . import run_mosp as base
    from .dataset import load
    from .eval_baselines import digest,file_hash,jsonl,verify_protocol_sources,write_json
    output=Path(output);directory=output/'data';directory.mkdir(exist_ok=True)
    manifest=directory/'manifest.json'
    pools=stage_datasets(output,settings)
    spec=dict(graph_curriculum=settings['graph_curriculum'],pools=pools,
              validation_graphs=settings['validation_graphs'],data_seed=settings['data_seed'],
              k=max(settings['curriculum']+settings['eval_ks']),max_candidates=settings['max_candidates'])
    sources={name:file_hash(Path(__file__).parent/name) for name in
             ('graph_curriculum.py','run_mosp.py','dataset.py','generate_graph_world.py','mosp.py','oracle.py','graph_images.py')}
    sources['../generate_graph_world.py']=file_hash(Path(__file__).parent.parent/'generate_graph_world.py')
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        error='Saved graph curriculum data changed; choose a fresh run'
        saved_sources=saved['generation_sources']
        if (saved['spec']!=spec or saved_sources.keys()!=sources.keys()
                or any(file_hash(directory/name)!=value for name,value in saved['files'].items())):
            raise ValueError(error)
        verify_protocol_sources(saved_sources,error)
        return
    if any(directory.glob('*.jsonl')):
        raise ValueError('Incomplete existing graph curriculum data; choose a fresh run')
    prior=[];training=[];final_validation=None
    for i,pool in enumerate(pools):
        stage_root=Path(pool['dataset']).parent.parent
        stage_root.mkdir(parents=True,exist_ok=True)
        stage_settings=deepcopy(settings)
        stage_settings.update(train_graphs=pool['count'],
            validation_graphs=settings['validation_graphs'] if i==len(pools)-1 else 0,
            data_seed=settings['data_seed']+i*1_000_003)
        print(f'Graph curriculum stage {i+1}/{len(pools)} | training graphs={pool["count"]} | '
              f'{json.dumps(pool["graph_generation"],sort_keys=True)}',flush=True)
        try:
            base.make_data(stage_root,stage_settings,pool['graph_generation'],exclusions=prior)
        except RuntimeError as error:
            raise RuntimeError(f'Graph curriculum stage {i+1}: {error}. Check --max-paths and graph complexity.') from error
        items=load(pool['dataset'])
        training.extend(items);prior.extend(items)
        if i==len(pools)-1:final_validation=stage_root/'data/validation.jsonl'
    jsonl(directory/'train.jsonl',[item.record() for item in training])
    (directory/'validation.jsonl').write_bytes(final_validation.read_bytes())
    files={str(path.relative_to(directory)):file_hash(path) for path in sorted(directory.rglob('*'))
           if path.is_file() and path!=manifest}
    write_json(manifest,dict(spec=spec,generation_sources=sources,files=files,
        statistics=dict(train_graphs=len(training),validation_graphs=settings['validation_graphs'],
                        stage_instance_ids=[[digest(item.record()) for item in load(p['dataset'])] for p in pools])))


class TrainingPools:
    def __init__(self,config,instances):
        from .dataset import load
        from .eval_baselines import digest
        self.config=config
        payload=config['graph_curriculum']
        stages=validate_parameters(payload['parameters'])
        if stages!=payload['stages']:
            raise ValueError('Graph curriculum stage metadata does not match its parameters')
        datasets=config['graph_curriculum_datasets']
        if len(stages)!=len(datasets) or config['steps']<max(len(stages),len(config['curriculum'])):
            raise ValueError('Every graph and k curriculum stage needs at least one training step')
        self.replay_probability=payload['replay_probability']
        if not 0<=self.replay_probability<1:
            raise ValueError('Invalid graph curriculum replay probability')
        # Policy randomness cannot change this schedule. Paired training seeds
        # across conditions receive identical scheduled stages/rehearsal choices.
        # Adaptive within-pool ordering can differ with observed reward variance.
        self.rng=random.Random(config['seed'] ^ 0x47524150)
        catalog={digest(item.record()):item for item in instances}
        self.pools=[];seen=set()
        for i,stage in enumerate(datasets):
            ids=[digest(item.record()) for item in load(stage['dataset'])]
            if stage['stage']!=i or not ids or len(set(ids))!=len(ids) or set(ids)&seen or set(ids)-catalog.keys():
                raise ValueError('Invalid, duplicate or mismatched graph curriculum pool membership')
            self.pools.append([catalog[key] for key in ids]);seen.update(ids)
        if seen!=catalog.keys():
            raise ValueError('Graph curriculum pools must partition the training dataset')
        if any(len(item.pareto)<max(config['curriculum']) for item in instances):
            raise ValueError('All graph pools must support every requested training k for rehearsal')

        from .graph_sampling import GraphSampler
        self.samplers=[GraphSampler(pool,config,config["seed"] ^ (0x53414D50+i)) for i,pool in enumerate(self.pools)]

    def observe(self,instance,rewards,terminal_scores):
        return self.samplers[self.last_pool].observe(instance,rewards,terminal_scores)

    def state(self):
        return dict(rng=self.rng.getstate(),pools=[sampler.state() for sampler in self.samplers])

    def audit(self):
        return dict(pools=[sampler.audit() for sampler in self.samplers])

    def choose(self,step):
        count=len(self.pools);steps=self.config['steps']
        scheduled=min(count-1,step*count//steps)
        previous=min(count-1,(step-1)*count//steps) if step else -1
        chosen=scheduled
        # Force the first group of each stage to its current pool. Later groups
        # use 80% current-stage graphs and 20% uniform earlier-stage rehearsal.
        if scheduled>0 and scheduled==previous and self.rng.random()<self.replay_probability:
            chosen=self.rng.randrange(scheduled)
        self.last_pool=chosen
        item,info=self.samplers[chosen].choose()
        return item,dict(**info,graph_stage=scheduled,graph_pool_stage=chosen,
            graph_stage_count=count,rehearsal=chosen!=scheduled)
