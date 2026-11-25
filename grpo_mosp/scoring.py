"""Terminal-only global Pareto scoring and diagnostic metrics."""
from .oracle import path_cost, hypervolume


def score(instance, archive, submission, k, objective='pareto'):
    slots = submission.get('paths') if isinstance(submission, dict) and set(submission)=={'paths'} else None
    wellformed = isinstance(slots,list) and len(slots)==k and all(isinstance(p,list) and all(isinstance(v,str) for v in p) for p in slots)
    def valid(p):
        return bool(p) and p[0]==instance.source and p[-1]==instance.target and len(set(p))==len(p) and all(instance.graph.has_edge(u,v) for u,v in zip(p,p[1:]))
    directed = [tuple(p) for p in slots if valid(p)] if wellformed else []
    submitted = set(directed) & set(archive)
    correct = submitted & instance.pareto
    discovered = set(archive) & instance.pareto
    reference = [(len(instance.graph)-1)*instance.max_cost+1]*len(instance.names)
    hv = hypervolume([path_cost(instance,p) for p in submitted], reference)
    denominator = hypervolume([path_cost(instance,p) for p in instance.pareto], reference)
    normalized = hv/denominator if denominator else 0.0
    return dict(reward=len(correct)/k if objective=='pareto' else normalized,
        valid_slot_fraction=len(directed)/k, distinct_valid_fraction=len(set(directed))/k,
        pareto_fraction=len(correct)/k, all_correct=float(len(correct)==k),
        discovered_pareto=len(discovered), discovery_fraction=min(k,len(discovered))/k,
        selection_gap=(min(k,len(discovered))-len(correct))/k,
        hypervolume=hv, normalized_hypervolume=normalized,
        nondominated_cost_vectors=len({path_cost(instance,p) for p in submitted if not any(
            all(a<=b for a,b in zip(path_cost(instance,q),path_cost(instance,p))) and path_cost(instance,q)!=path_cost(instance,p) for q in submitted)}))
