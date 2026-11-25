"""Evaluate graph-disjoint tasks and save per-task as well as aggregate metrics."""
import argparse
import json
from pathlib import Path
from .model_options import add_thinking_arguments, apply_thinking_arguments
from .baselines import baseline
from .dataset import load
from .scoring import score


def metrics(env):
    result = score(env._instance,env.archive,env.submission,env.k,env.objective)
    result.update(protocol_failure=float(env.failure), dead_end_rate=env.dead_attempts/max(1,env.attempts),
                  edge_traversals=env.moves)
    return result


def evaluate(instances,k=3,policy=None,objective='pareto'):
    rows = []
    for item in instances:
        for method in ['random','memory_pareto_filter','privileged_selection','full_information']+(['qwen'] if policy else []):
            if method=='qwen':
                from .rollout import collect
                result = metrics(collect(policy,item,k,objective=objective).env)
            elif method=='full_information':
                # This solver knows the entire frontier and has no exploration budget.
                paths = sorted(item.pareto)[:k]
                result = score(item,{p:None for p in paths},dict(paths=[list(p) for p in paths]),k,objective)
            else:
                result = metrics(baseline(item,k,method,item.seed,objective=objective))
            rows.append(dict(seed=item.seed,method=method,**result))
    summary = {}
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method']==method]
        summary[method] = {key:sum(r[key] for r in subset)/len(subset) for key in subset[0] if key not in ('method','seed')}
    return dict(summary=summary,tasks=rows)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--output',default='jm_test/graph_world/grpo_mosp/evaluation.json')
    parser.add_argument('--config')
    parser.add_argument('--adapter')
    parser.add_argument('--objective',choices=['pareto','hypervolume'])
    add_thinking_arguments(parser)
    args = parser.parse_args()
    policy = None
    objective = args.objective or 'pareto'
    if args.config:
        import torch
        from .policy import load_policy
        config = apply_thinking_arguments(json.loads(Path(args.config).read_text()), args)
        objective = args.objective or config['objective']
        torch.manual_seed(config['seed'])
        policy = load_policy(config,args.adapter,training=False)
    report = evaluate(load(args.dataset),args.k,policy,objective)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2))
