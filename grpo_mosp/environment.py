"""Deterministic finite-horizon exploration; complete tasks are episodes."""
import json
from .memory import serialize
from .scoring import score


class Environment:
    def __init__(self, instance, k=1, budget=None, objective='pareto'):
        self._instance = instance  # Evaluator only: the policy receives serialized observations.
        self.source, self.target = instance.source, instance.target
        self.names, self.max_cost, self.n = instance.names, instance.max_cost, len(instance.graph)
        self.k, self.budget = k, 8*k if budget is None else budget
        if k < 1 or self.budget < k or len(instance.pareto) < k:
            raise ValueError('Require k>=1, B>=k and at least k true Pareto paths')
        if objective not in ('pareto','hypervolume'):
            raise ValueError('Unknown objective')
        self.objective = objective
        self.attempts = self.moves = self.dead_attempts = 0
        self.archive, self.neighbors, self.known = {}, {}, {}
        self.dead_nodes = set()
        self.phase, self.failure, self.last_event = 'traversal', False, None
        self.submission, self.reward = None, 0.0
        self._reset()

    def _reveal(self):
        self.neighbors[self.current] = sorted(self._instance.graph.successors(self.current))

    def _reset(self):
        self.current, self.prefix, self.cost = self.source, [self.source], [0]*len(self.names)
        self._reveal()

    def observation(self):
        return serialize(self)

    def choices(self):
        if self.phase == 'traversal':
            return [dict(move_to=v) for v in self.neighbors[self.current]]
        return []

    def step(self, action):
        if self.phase == 'terminal':
            raise RuntimeError('Task already terminated')
        try:
            action = json.loads(action) if isinstance(action, str) else action
            if self.phase == 'submission':
                self.submission = action
                # Cache evaluator-only terminal statistics for reporting; serialize()
                # has an explicit allowlist and never exposes these to the policy.
                self._terminal_metrics = score(self._instance, self.archive, action, self.k, self.objective)
                self.reward = self._terminal_metrics['reward']
                # A syntactically bad final response is an explicit protocol failure.
                if not isinstance(action, dict) or set(action) != {'paths'} or not isinstance(action['paths'], list) or len(action['paths']) != self.k or any(not isinstance(p,list) or any(not isinstance(v,str) for v in p) for p in action['paths']):
                    self.failure = True
                self.phase = 'terminal'
                return self.reward
            if not isinstance(action,dict) or set(action) != {'move_to'} or action['move_to'] not in self.neighbors[self.current]:
                raise ValueError('Illegal movement')
            u, v = self.current, action['move_to']
            edge_cost = [self._instance.graph[u][v].get(name,1) for name in self.names]
            self.known[u,v] = edge_cost
            self.cost = [a+b for a,b in zip(self.cost, edge_cost)]
            self.current = v
            self.prefix.append(v)
            self.moves += 1
            self._reveal()
            self.last_event = dict(edge=[u,v], cost=edge_cost, prefix=self.prefix.copy(), total=self.cost.copy())
            if v == self.target or not self.neighbors[v]:
                success = v == self.target
                if success:
                    self.archive[tuple(self.prefix)] = self.cost.copy()
                else:
                    self.dead_nodes.add(v)
                    self.dead_attempts += 1
                self.last_event['outcome'] = 'target' if success else 'dead_end'
                self.attempts += 1
                if self.attempts == self.budget:
                    self.phase = 'submission'
                else:
                    self._reset()
            assert self.moves <= self.budget*(self.n-1)
            return 0.0  # No shaping, including at the end of an individual traversal.
        except (ValueError, TypeError, KeyError):
            self.phase, self.failure, self.reward = 'terminal', True, 0.0
            return 0.0
