"""Observable, task-deduplicated discovery bonuses; never expose oracle labels.

Bonuses are earned at edge/path discovery events, then paid only after a normal
final submission. The existing outcome-GRPO update uses their task-level sum.
"""
import math

SIGNALS = ('edge','path','hypervolume','cost_diversity','path_disjointness')
CONDITIONS = ('none','edge','path','both')


def signal_names(condition):
    if condition=='none':return set()
    if condition=='all':return set(SIGNALS)
    if condition=='both':return {'edge','path'}
    names=set(condition.replace('-', '_').split('+'))
    if not names or names-set(SIGNALS):raise ValueError(f'Unknown shaping condition: {condition}')
    return names


def condition_name(names):
    names=set(names)
    if not names:return 'none'
    if names=={'edge','path'}:return 'both'
    return '+'.join(name for name in SIGNALS if name in names)


def experiment_conditions(selection, signals=None):
    if selection=='ablation':return list(CONDITIONS)
    if selection=='factorial':
        from itertools import combinations
        names=list(SIGNALS if signals is None else signals)
        return [condition_name(part) for n in range(len(names)+1) for part in combinations(names,n)]
    return [condition_name(signal_names(selection))]


def options(config):
    result = dict(edge=False,path=False,edge_weight=.01,path_weight=.05,
                  edge_cap=0,path_cap=0,require_submission=True,
                  hypervolume=False,cost_diversity=False,path_disjointness=False,
                  hypervolume_weight=.05,cost_diversity_weight=.05,path_disjointness_weight=.05)
    if config is not None:
        if set(config)-set(result):
            raise ValueError(f'Unknown shaping options: {set(config)-set(result)}')
        result.update(config)
    for key in (*SIGNALS,'require_submission'):
        if type(result[key]) is not bool:
            raise ValueError(f'shaping.{key} must be a boolean')
    for key in (name+'_weight' for name in SIGNALS):
        value=result[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError(f'shaping.{key} must be finite and nonnegative')
    for key in ('edge_cap','path_cap'):
        if type(result[key]) is not int or result[key]<0:
            raise ValueError(f'shaping.{key} must be a nonnegative integer')
    return result


def for_condition(condition, **kwargs):
    selected=signal_names(condition)
    return options(dict(kwargs,**{name:name in selected for name in SIGNALS}))


class DiscoveryBonuses:
    def __init__(self, env, config):
        self.config=options(config)
        self.edge_cap=self.config['edge_cap'] or env.budget
        self.path_cap=self.config['path_cap'] or env.k
        self.edges,self.paths=set(env.known),set(env.archive)
        self.events=[]
        self.edge_total=self.path_total=0.
        self.extra_totals={name:0. for name in SIGNALS[2:]}
        self.vector_seen=set();self.path_edges=[];self.hv=0.
        # Public graph-size/cost bounds, never the true frontier volume.
        self.reference=[(env.n-1)*env.max_cost+1]*len(env.names)

    def observe(self, env, action_index):
        # Persistent sets span all attempts; reset-to-source never resets novelty.
        new_edges=set(env.known)-self.edges
        new_paths=set(env.archive)-self.paths
        old_e,old_p=len(self.edges),len(self.paths)
        self.edges.update(new_edges);self.paths.update(new_paths)
        edge=self.config['edge_weight']*(min(len(self.edges),self.edge_cap)-min(old_e,self.edge_cap))/self.edge_cap if self.config['edge'] else 0.
        path=self.config['path_weight']*(min(len(self.paths),self.path_cap)-min(old_p,self.path_cap))/self.path_cap if self.config['path'] else 0.
        self.edge_total+=edge;self.path_total+=path
        extra={name:0. for name in SIGNALS[2:]}
        for completed in sorted(new_paths):
            vector=tuple(env.archive[completed])
            if self.config['cost_diversity'] and self.vector_seen:
                # Nearest-vector mean L1 distance in a fixed unit box. Equal
                # vectors add zero, but their paths remain in the archive.
                distance=min(sum(min(1.,abs(a-b)/r) for a,b,r in zip(vector,other,self.reference))/len(vector)
                             for other in self.vector_seen)
                extra['cost_diversity']+=distance/max(1,env.k-1)
            self.vector_seen.add(vector)
            edges=set(zip(completed,completed[1:]))
            if self.config['path_disjointness'] and self.path_edges:
                distance=min(1-len(edges & other)/len(edges | other) for other in self.path_edges)
                extra['path_disjointness']+=distance/max(1,env.k-1)
            self.path_edges.append(edges)
        if new_paths and self.config['hypervolume']:
            from .oracle import hypervolume
            # Normalize against the fixed reference box rather than an oracle
            # frontier. Compute the union once per completed path, not per move.
            points=[tuple(max(0.,min(1.,v/r)) for v,r in zip(cost,self.reference))
                    for cost in env.archive.values()]
            volume=hypervolume(points,[1.]*len(self.reference))
            extra['hypervolume']=max(0.,volume-self.hv)
            self.hv=volume
        for name,increment in extra.items():
            weight=self.config[name+'_weight']
            paid=min(max(0.,weight-self.extra_totals[name]),weight*increment)
            self.extra_totals[name]+=paid
            extra[name]=paid
        if new_edges or new_paths:
            self.events.append(dict(action_index=action_index,new_edges=[list(e) for e in sorted(new_edges)],
                new_paths=[list(p) for p in sorted(new_paths)],edge_bonus_earned=edge,path_bonus_earned=path,
                **{name+'_bonus_earned':value for name,value in extra.items()}))

    def finish(self, rollout):
        env=rollout.env
        eligible=rollout.status=='ok' and not env.failure and env.phase=='terminal'
        # The constrained grammar permits empty slots. If paths are available,
        # require submission of min(k, archive size) distinct archived paths so
        # the model cannot collect bonuses and evade final selection. No Pareto
        # membership is used: dominated but valid paths remain eligible.
        slots=env.submission.get('paths',[]) if isinstance(env.submission,dict) else []
        valid_slots=isinstance(slots,list) and len(slots)==env.k and all(isinstance(p,list) and all(isinstance(v,str) for v in p) for p in slots)
        submitted={tuple(p) for p in slots if p} & set(env.archive) if valid_slots else set()
        if self.config['require_submission']:
            eligible=eligible and valid_slots and len(submitted)>=min(env.k,len(env.archive))
        edge=self.edge_total if eligible else 0.
        path=self.path_total if eligible else 0.
        extras={name+'_bonus':value if eligible else 0. for name,value in self.extra_totals.items()}
        combined=edge+path+sum(extras.values())
        terminal=rollout.reward
        rollout.shaping=dict(terminal_reward=terminal,edge_bonus=edge,path_bonus=path,
            combined_reward=terminal+combined,eligible=eligible,**extras,
            unique_edges=len(self.edges),unique_paths=len(self.paths),edge_cap=self.edge_cap,path_cap=self.path_cap,
            events=self.events)
        # env.reward and observations stay unshaped; only the trainer consumes
        # this augmented scalar. Failed/limited rollouts receive zero bonuses.
        rollout.reward=terminal+combined
        return rollout
