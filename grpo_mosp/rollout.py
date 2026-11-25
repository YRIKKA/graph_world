"""Collect isolated complete tasks while all policy parameters remain frozen."""
from dataclasses import dataclass
from .environment import Environment
from .memory import RULES
from .policy import ContextOverflow


@dataclass
class Rollout:
    decisions: list
    reward: float
    env: Environment
    messages: list
    status: str = 'ok'
    limit_details: dict | None = None
    shaping: dict | None = None


def collect(policy, instance, k, budget=None, objective='pareto', shaping=None):
    env = Environment(instance, k, budget, objective)
    messages = [dict(role='system',content=RULES)]
    decisions = []
    tracker = None
    if shaping is not None:
        from .shaping import DiscoveryBonuses
        tracker = DiscoveryBonuses(env,shaping)
    action_index=0
    while env.phase != 'terminal':
        messages.append(dict(role='user',content=env.observation()))
        try:
            action, parts = policy.action(messages, env)
        except ContextOverflow as error:
            # Keep this repetition/group member with terminal reward zero. Never
            # discard long tasks, retry until success, or truncate their history.
            # Evaluation already uses the same intention-to-evaluate limit rule.
            decisions.extend(getattr(policy,'pending_decisions',[]))
            env.phase,env.reward = 'terminal',0.0
            result = Rollout(decisions,0.0,env,messages,'context_limit',
                dict(tokens=error.length,limit=error.limit,moves=env.moves,
                     attempts_completed=env.attempts,budget=env.budget))
            return tracker.finish(result) if tracker else result
        decisions.extend(parts)
        messages.append(dict(role='assistant',content=action))
        env.step(action)
        if tracker:
            tracker.observe(env,action_index)
        action_index+=1
    result = Rollout(decisions, env.reward, env, messages)
    return tracker.finish(result) if tracker else result


def group(policy, instance, k, size=8, budget=None, objective='pareto', shaping=None):
    # Each collect constructs fresh memory; no parameters change inside the group.
    return [collect(policy, instance, k, budget, objective, shaping) for _ in range(size)]
