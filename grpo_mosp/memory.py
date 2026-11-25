"""Deterministic, complete observable memory; no evaluator data is serialized."""
import json

RULES = """Discover Pareto-minimal paths; minimize every named additive cost independently.
Only outgoing neighbors are legal moves. Costs are revealed only after traversal.
Use exactly {"move_to":"node_id"} during exploration. Target or dead-end arrival
ends an attempt and automatically resets to source. Memory persists. After all
attempts submit {"paths":[...]} with exactly k slots. Each slot must be an archived
node sequence or []. Duplicate paths earn no additional credit. Only the final
submission earns reward. Do not infer optimality from the archive alone."""


def serialize(env):
    # Explicit allowlist: never serialize vars(env) or the hidden instance.
    return json.dumps(dict(source=env.source, target=env.target, k=env.k, B=env.budget,
        d=len(env.names), W=env.max_cost, n=env.n, cost_names=env.names,
        phase=env.phase, current=env.current, prefix=env.prefix, cost=env.cost,
        attempts_completed=env.attempts, attempts_remaining=env.budget-env.attempts,
        neighbors=env.neighbors, known_costs=[dict(edge=list(e),cost=c) for e,c in sorted(env.known.items())],
        dead_ends=sorted(env.dead_nodes), archive=[dict(path=list(p),cost=c) for p,c in sorted(env.archive.items())],
        last_event=env.last_event), sort_keys=True, separators=(',', ':'))
