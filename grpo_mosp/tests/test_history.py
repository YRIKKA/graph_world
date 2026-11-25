"""Regression checks for lossless memory compression and bounded-context tasks."""
import json
from copy import deepcopy

import pytest

from ..environment import Environment
from ..history import apply_changes, changes, compact_history, expand_history
from ..memory import RULES
from .helpers import fixture


def long_history(attempts=24):
    env = Environment(fixture(),k=2,budget=attempts)
    messages = [dict(role='system',content=RULES)]
    # Repeated routes plus distinct/tied/dominated discoveries exercise archives,
    # source resets, dead-end arrivals, and insertions into sorted memory lists.
    for i in range(attempts):
        branch = ('a','b','c','z')[i%4]
        for v in ([branch] if branch=='z' else [branch,'t']):
            messages.append(dict(role='user',content=env.observation()))
            action = json.dumps(dict(move_to=v),separators=(',',':'))
            messages.append(dict(role='assistant',content=action))
            env.step(action)
    messages.append(dict(role='user',content=env.observation()))
    return messages


def test_delta_roundtrip_preserves_every_turn_and_complete_current_memory():
    messages = long_history()
    original = deepcopy(messages)
    encoded = compact_history(messages)
    assert expand_history(encoded) == messages
    assert messages == original  # Raw rollout/audit messages are never modified.
    assert encoded[-1] == messages[-1]
    assert [m for m in encoded if m['role']=='assistant'] == [m for m in messages if m['role']=='assistant']
    assert sum(len(m['content']) for m in encoded) < .7*sum(len(m['content']) for m in messages)


@pytest.mark.parametrize('old,new',[
    ({'x':[1,2,3,4]},{'x':[1,7,8,4],'z':1}),
    ({'x':[1,2,3],'delete':4},{'x':[1,3]}),
    ({'x':[]},{'x':[{'a':1}]}),
    ([1,2],[1,2,3]),([],[]),(1,{'x':2}),
])
def test_patches_roundtrip_insertions_removals_replacements(old,new):
    untouched = deepcopy(old)
    assert apply_changes(old,changes(old,new)) == new
    assert old == untouched


def test_non_environment_messages_are_preserved():
    messages = [dict(role='system',content='System'),dict(role='user',content='Ready.')]
    assert compact_history(messages) == messages


def test_compaction_prevents_the_repeated_memory_overflow():
    pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    from .helpers import ByteTokenizer,tiny_model
    messages = long_history(attempts=40)
    full = ConstrainedPolicy(tiny_model(),ByteTokenizer(),history_format='full')
    delta = ConstrainedPolicy(tiny_model(),ByteTokenizer(),history_format='delta')
    full_length,delta_length = len(full.prompt(messages)),len(delta.prompt(messages))
    assert full_length > 32768
    assert delta_length < 32768
    delta.check_length(delta_length)


def test_context_exhaustion_keeps_every_group_member_and_zero_reward():
    torch = pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    from ..rollout import group
    from ..grpo import update
    from .helpers import ByteTokenizer,tiny_model
    model = tiny_model()
    # Deliberately impossible limit: every member must remain in the group,
    # terminate explicitly, and contribute zero reward; no retry or filtering.
    policy = ConstrainedPolicy(model,ByteTokenizer(),context_limit=2,history_format='delta')
    tasks = group(policy,fixture(),k=1,size=3,budget=8)
    assert len(tasks)==3
    assert all(r.status=='context_limit' and r.reward==0 and r.env.phase=='terminal' for r in tasks)
    assert all(r.limit_details['tokens']>2 and not r.env.failure for r in tasks)
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    stats = update(policy,tasks,optimizer,dict(beta=0,clip=.2,epochs_per_group=1,max_grad_norm=1))
    assert stats['zero_variance_group'] == 1 and torch.equal(before,model.weight)


def test_state_actions_reconstructs_every_intermediate_observation():
    from ..history import compact_state_actions,expand_state_actions
    messages = long_history(attempts=12)
    for end in range(2,len(messages)+1,2):
        original = messages[:end]
        encoded = compact_state_actions(original)
        assert len(encoded)==2
        assert expand_state_actions(encoded)==original
        packed = json.loads(encoded[-1]['content'])
        assert packed['memory']==json.loads(original[-1]['content'])
        assert len(packed['move_history'])==(len(original)-2)//2


def test_state_actions_handles_long_tasks_without_losing_information():
    pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy
    from ..history import compact_state_actions,expand_state_actions
    from .helpers import ByteTokenizer,tiny_model
    messages = long_history(attempts=80)
    assert sum(len(m['content']) for m in messages)>32768
    policy = ConstrainedPolicy(tiny_model(),ByteTokenizer(),history_format='state_actions')
    assert len(policy.prompt(messages))<32768
    assert expand_state_actions(compact_state_actions(messages))==messages


def test_state_actions_preserves_new_or_unrecognized_feedback_via_full_fallback():
    from ..history import compact_state_actions
    messages = long_history(attempts=3)
    observation = json.loads(messages[3]['content'])
    observation['new_feedback_field'] = 'must not be dropped'
    messages[3]['content']=json.dumps(observation,sort_keys=True,separators=(',',':'))
    assert compact_state_actions(messages)==messages


def test_training_and_both_evaluation_arms_use_identical_compact_prompts(monkeypatch):
    pytest.importorskip('torch')
    from .. import model_backends
    from ..model_defaults import DEFAULT_MODEL,DEFAULT_REVISION,initial_checkpoint
    from ..policy import load_policy
    from ..eval_baselines import load_frozen,limited_policy
    from .helpers import ByteTokenizer,tiny_model
    monkeypatch.setattr(model_backends,'load_model',lambda *a,**kw:(tiny_model(),ByteTokenizer()))
    config=dict(model=DEFAULT_MODEL,revision=DEFAULT_REVISION,context_limit=32768,
                temperature=1.,history_format='state_actions')
    trained=load_policy(config)
    frozen=load_frozen(initial_checkpoint(config),config)
    post=load_frozen(dict(initial_checkpoint(config),adapter='test-adapter'),config)
    messages=long_history()
    assert trained.prompt(messages)==frozen.prompt(messages)==limited_policy(post,config).prompt(messages)


def test_limit_on_later_submission_slot_preserves_already_generated_tokens():
    pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy,ContextOverflow
    from ..rollout import collect
    from .helpers import ByteTokenizer,tiny_model
    class SecondSlotOverflow(ConstrainedPolicy):
        def action(self,messages,env):
            self.phase=env.phase
            return super().action(messages,env)
        def sample(self,context,candidates):
            if self.phase=='submission':
                self.slot_count=getattr(self,'slot_count',0)+1
                if self.slot_count==2:
                    raise ContextOverflow(self.context_limit+1,self.context_limit)
            return super().sample(context,candidates)
    policy=SecondSlotOverflow(tiny_model(),ByteTokenizer(),context_limit=50000,history_format='state_actions')
    task=collect(policy,fixture(),k=2,budget=2)
    assert task.status=='context_limit'
    assert len(policy.pending_decisions)==1
    assert task.decisions[-1] is policy.pending_decisions[0]
    assert policy.tokenizer.decode(task.decisions[-1].tokens).startswith('{"paths":[')


@pytest.mark.parametrize('batched',[False,True])
def test_context_checks_count_token_ids_in_transformers_batch_encoding(batched):
    pytest.importorskip('torch')
    from ..policy import ConstrainedPolicy,ContextOverflow
    from .helpers import tiny_model
    class MappingTokenizer:
        def apply_chat_template(self,*args,**kwargs):
            ids=[1,2,3,4,5]
            return {'input_ids':[ids] if batched else ids,'attention_mask':[1]*5}
    policy=ConstrainedPolicy(tiny_model(),MappingTokenizer(),context_limit=4)
    tokens=policy.prompt([dict(role='user',content='Hello')])
    assert tokens==[1,2,3,4,5]
    with pytest.raises(ContextOverflow):
        policy.check_length(len(tokens))
