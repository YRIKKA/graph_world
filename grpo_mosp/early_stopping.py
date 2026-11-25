"""Fixed validation-only checkpoint monitoring, isolated from training randomness."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from .dataset import load
from .reporting import Progress


class Plateau:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best = self.reference = float('-inf')
        self.best_step = self.stale = 0

    def observe(self, value, step):
        if not np.isfinite(value):
            raise ValueError('Non-finite validation score')
        # Always export the actual best score, even when its gain is smaller
        # than min_delta. Only significant improvement resets patience.
        improved = value > self.best
        if improved:
            self.best, self.best_step = value, step
        if value > self.reference + self.min_delta:
            self.reference, self.stale = value, 0
        else:
            self.stale += 1
        return improved


@contextmanager
def validation_mode(policy):
    """Validation must not perturb subsequent graph selection or action samples."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module,module.training) for module in policy.model.modules()]
    try:
        policy.model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module,mode in modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def validation_instances(config, training):
    from .eval_baselines import IsomorphismIndex, digest
    path = config.get('validation_dataset')
    if not path:
        raise ValueError('early_stopping=true requires validation_dataset; supply --validation-dataset or disable early stopping')
    instances = load(path)
    ks = config.get('validation_ks',list(dict.fromkeys(config['curriculum'])))
    if not ks or any(type(k) is not int or k<1 for k in ks):
        raise ValueError('validation_ks must contain positive integers')
    if not instances or any(len(item.pareto)<max(ks) for item in instances):
        raise ValueError('Validation needs nonempty data with enough Pareto paths for every validation k')
    # Reject training graphs and relabeled copies BEFORE loading the GPU model.
    # Also reject duplicates in validation rather than overweighting a family.
    index = IsomorphismIndex()
    for item in training:
        index.add_if_new(item)
    for item in instances:
        if not index.add_if_new(item):
            raise ValueError('Validation overlaps training or contains duplicate graph copies')
    return sorted(instances,key=lambda item:digest(item.record()))[:config['early_stopping_graphs']]


def validate(policy, instances, config):
    from .eval_baselines import evaluate_task, digest
    ks = config.get('validation_ks',list(dict.fromkeys(config['curriculum'])))
    controls = dict(config,device=str(policy.device),max_action_tokens=config.get('max_action_tokens',512),
                    max_task_tokens=config.get('max_task_tokens',16384))
    progress = Progress('Early stopping | validation tasks',len(instances)*len(ks)*config['early_stopping_repetitions'])
    tasks = []
    with validation_mode(policy):
        for instance in instances:
            graph_id = digest(instance.record())
            for k in ks:
                for repetition in range(config['early_stopping_repetitions']):
                    # Common random numbers across checks AND training runs;
                    # each task starts fresh memory and receives exactly B=8k.
                    seed = int(digest(['early-stopping',config['early_stopping_seed'],graph_id,k,repetition])[:15],16)
                    result = evaluate_task(policy,instance,k,8,seed,controls)
                    tasks.append(dict(graph_id=graph_id,k=k,repetition=repetition,seed=seed,
                                      status=result['status'],metrics=result['metrics']))
                    progress.advance()
    return dict(mean_S=sum(t['metrics']['S'] for t in tasks)/len(tasks),tasks=tasks)
