"""Lossless history representations with exact reconstruction for auditing.

The default keeps complete current memory and every prior move. The optional
JSON-delta format keeps every chat turn while replacing repeated old tables by
patches. Both retain all observable information; original messages stay in raw
rollouts. No graph facts, failed attempts, actions or archive paths are truncated.
"""
from copy import deepcopy
import json

DELTA_RULES = '\nEarlier user messages may contain {"memory_delta":[...]}. Apply each operation to the previous observation: ["s",path,value] sets a field, ["d",path] deletes a field, ["a",path,start,count,items] splices a list. Paths are lists of keys/indices. Full observations replace the table. The final user message always contains the complete current memory.'


def changes(old, new, path=()):
    """Encode exact changes; common list prefixes/suffixes are never repeated."""
    if old == new:
        return []
    if isinstance(old,dict) and isinstance(new,dict):
        result = [['d',[*path,key]] for key in sorted(old.keys()-new.keys())]
        for key in sorted(new):
            child = (*path,key)
            result.extend(changes(old[key],new[key],child) if key in old else [['s',list(child),new[key]]])
        return result
    if isinstance(old,list) and isinstance(new,list):
        start = 0
        while start < min(len(old),len(new)) and old[start] == new[start]:
            start += 1
        end = 0
        while end < min(len(old),len(new))-start and old[len(old)-end-1] == new[len(new)-end-1]:
            end += 1
        return [['a',list(path),start,len(old)-start-end,new[start:len(new)-end if end else len(new)]]]
    return [['s',list(path),new]]


def apply_changes(previous, operations):
    """Inverse used by audit tools/tests to reconstruct all original memories."""
    state = deepcopy(previous)
    for op,path,*args in operations:
        if not path:
            if op == 's':
                state = deepcopy(args[0])
            elif op == 'a':
                start,count,items = args
                state[start:start+count] = deepcopy(items)
            else:
                raise ValueError('Cannot delete the root observation')
            continue
        parent = state
        for key in path[:-1]:
            parent = parent[key]
        key = path[-1]
        if op == 's':
            parent[key] = deepcopy(args[0])
        elif op == 'd':
            del parent[key]
        elif op == 'a':
            start,count,items = args
            parent[key][start:start+count] = deepcopy(items)
        else:
            raise ValueError(f'Unknown memory delta operation: {op}')
    return state


def observation(message):
    if message.get('role') != 'user':
        return None
    try:
        value = json.loads(message['content'])
    except (ValueError,TypeError):
        return None
    # Only compact environment tables, never arbitrary user prose/transcripts.
    return value if isinstance(value,dict) and {'source','target','neighbors','known_costs','archive','prefix'} <= value.keys() else None


def compact_history(messages):
    result,previous,used = [],None,False
    for i,message in enumerate(messages):
        value = observation(message)
        content = message['content']
        if value is not None:
            if previous is not None and i != len(messages)-1:
                encoded = json.dumps(dict(memory_delta=changes(previous,value)),separators=(',',':'))
                # A large change can be cheaper to represent as a full snapshot.
                if len(encoded) < len(content):
                    content,used = encoded,True
            previous = value
        result.append(dict(message,content=content))
    if used:
        if result[0]['role'] == 'system':
            result[0]['content'] += DELTA_RULES
        else:
            result.insert(0,dict(role='system',content=DELTA_RULES.strip()))
    return result


def expand_history(messages):
    """Recover every original environment observation from the compact prompt."""
    result,previous = [],None
    for message in messages:
        content = message['content']
        if message['role'] == 'system' and content.endswith(DELTA_RULES):
            content = content[:-len(DELTA_RULES)]
        if message['role'] == 'user':
            try:
                value = json.loads(content)
            except (ValueError,TypeError):
                value = None
            if isinstance(value,dict) and set(value) == {'memory_delta'}:
                if previous is None:
                    raise ValueError('Memory delta has no preceding full observation')
                previous = apply_changes(previous,value['memory_delta'])
                content = json.dumps(previous,sort_keys=True,separators=(',',':'))
            elif observation(message) is not None:
                previous = value
        result.append(dict(message,content=content))
    return result


STATE_ACTION_RULES = '\nThe user supplies memory (the complete current observation) and move_history (every prior move destination in order). Earlier observations are represented losslessly: replay these moves from source, revealing neighbors on arrival and edge costs on traversal; target/dead-end arrivals reset to source while attempts remain. All discovered costs, adjacencies, dead ends and completed paths remain in memory.'


def observations_from_moves(memory, moves):
    """Reconstruct ALL observations using observable facts only, without a graph.

    A fixed deterministic graph reveals a node's same adjacency on every visit
    and an edge's same cost on every traversal. The complete current table and
    chronological actions therefore determine precisely what was known earlier.
    This inverse also guards compaction against future observation-schema changes.
    """
    state = {k:deepcopy(memory[k]) for k in ('source','target','k','B','d','W','n','cost_names')}
    source,target,budget = state['source'],state['target'],state['B']
    known,archive,dead = {},{},set()
    costs = {tuple(row['edge']):row['cost'] for row in memory['known_costs']}
    state.update(phase='traversal',current=source,prefix=[source],cost=[0]*state['d'],
        attempts_completed=0,attempts_remaining=budget,neighbors={source:memory['neighbors'][source]},
        known_costs=[],dead_ends=[],archive=[],last_event=None)
    yield deepcopy(state)
    for destination in moves:
        if state['phase'] != 'traversal' or destination not in state['neighbors'][state['current']]:
            raise ValueError('Move history is incompatible with observable memory')
        edge = (state['current'],destination)
        cost = costs[edge]
        known[edge] = cost
        state['current'] = destination
        state['prefix'] = [*state['prefix'],destination]
        state['cost'] = [a+b for a,b in zip(state['cost'],cost)]
        state['neighbors'][destination] = memory['neighbors'][destination]
        event = dict(edge=list(edge),cost=cost,prefix=state['prefix'].copy(),total=state['cost'].copy())
        if destination == target or not state['neighbors'][destination]:
            success = destination == target
            if success:
                archive[tuple(state['prefix'])] = state['cost'].copy()
            else:
                dead.add(destination)
            event['outcome'] = 'target' if success else 'dead_end'
            state['attempts_completed'] += 1
            state['attempts_remaining'] -= 1
            if state['attempts_completed'] == budget:
                state['phase'] = 'submission'
            else:
                state.update(current=source,prefix=[source],cost=[0]*state['d'])
        state.update(last_event=event,
            known_costs=[dict(edge=list(e),cost=c) for e,c in sorted(known.items())],
            archive=[dict(path=list(p),cost=c) for p,c in sorted(archive.items())],dead_ends=sorted(dead))
        yield deepcopy(state)


def compact_state_actions(messages):
    """Keep full current memory and every move; verify exact reversibility first.

    The guard deliberately falls back to raw history for arbitrary chat messages
    or a future environment schema that the reconstruction cannot reproduce. It
    never silently discards new fields, intermediate feedback or assistant text.
    """
    if len(messages)<2 or messages[0]['role']!='system' or messages[-1]['role']!='user':
        return messages
    memory = observation(messages[-1])
    if memory is None or len(messages)%2:
        return messages
    moves = []
    try:
        for i in range(2,len(messages),2):
            action = json.loads(messages[i]['content'])
            if messages[i]['role']!='assistant' or set(action)!={'move_to'}:
                return messages
            moves.append(action['move_to'])
            # Preserve formatting and any extra message fields via the fallback.
            if messages[i] != dict(role='assistant',content=json.dumps(action,separators=(',',':'))):
                return messages
        snapshots = observations_from_moves(memory,moves)
        for original,reconstructed in zip(messages[1::2],snapshots,strict=True):
            if original != dict(role='user',content=json.dumps(reconstructed,sort_keys=True,separators=(',',':'))):
                return messages
    except (KeyError,TypeError,ValueError):
        return messages
    return [dict(messages[0],content=messages[0]['content']+STATE_ACTION_RULES),
            dict(role='user',content=json.dumps(dict(memory=memory,move_history=moves),sort_keys=True,separators=(',',':')))]


def expand_state_actions(messages):
    """Audit inverse for the compact memory/move representation."""
    if len(messages)!=2 or not messages[0]['content'].endswith(STATE_ACTION_RULES):
        return messages
    record = json.loads(messages[1]['content'])
    moves = record['move_history']
    result = [dict(messages[0],content=messages[0]['content'][:-len(STATE_ACTION_RULES)])]
    for i,state in enumerate(observations_from_moves(record['memory'],moves)):
        if i:
            result.append(dict(role='assistant',content=json.dumps(dict(move_to=moves[i-1]),separators=(',',':'))))
        result.append(dict(role='user',content=json.dumps(state,sort_keys=True,separators=(',',':'))))
    return result
