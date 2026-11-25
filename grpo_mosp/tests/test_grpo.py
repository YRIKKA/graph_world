import pytest
torch=pytest.importorskip('torch')
from ..policy import ConstrainedPolicy, masked_logps
from ..grpo import advantages, surrogate, update
from ..rollout import group
from .helpers import tiny_model, ByteTokenizer, fixture


def test_zero_variance_and_loss_mask():
    adv,zero=advantages([.5,.5])
    assert zero and not adv.any() and not adv.requires_grad
    current=torch.zeros(3,requires_grad=True)
    old=torch.zeros(3)
    loss=surrogate(current,old,old,adv[0],beta=0).mean()
    loss.backward()
    assert torch.equal(current.grad,torch.zeros(3))
    adv,zero=advantages([0,1])
    assert not zero and torch.allclose(adv,torch.tensor([-1.,1.]))


def test_constrained_probabilities_and_environment_mask():
    torch.manual_seed(3)
    model=tiny_model()
    policy=ConstrainedPolicy(model,ByteTokenizer(),context_limit=10000)
    d=policy.sample([250,251],['{"move_to":"a"}','{"move_to":"b"}'])
    replay=policy.replay(d)
    assert len(replay)==len(d.tokens)  # Only output tokens, no context positions.
    assert torch.allclose(replay.detach(),torch.tensor(d.old_logps),atol=1e-6)
    for token,allowed in zip(d.tokens,d.allowed): assert token in allowed
    assert sum(len(a)>1 for a in d.allowed)==1
    (-replay.sum()).backward()
    assert model.weight.grad[250]==0 and model.weight.grad[251]==0
    assert torch.allclose(masked_logps(torch.zeros(256),[97,98]).exp(),torch.tensor([.5,.5]))


def test_context_overflow_never_truncates():
    policy=ConstrainedPolicy(tiny_model(),ByteTokenizer(),context_limit=2)
    with pytest.raises(RuntimeError,match='Context overflow'): policy.sample([1,2],['abc'])


@pytest.mark.parametrize('epochs',[1,2])
def test_small_training_smoke(epochs):
    torch.manual_seed(2)
    model=tiny_model()
    policy=ConstrainedPolicy(model,ByteTokenizer(),context_limit=50000)
    tasks=group(policy,fixture(),k=1,size=8,budget=1)
    assert all(not r.env.failure for r in tasks)
    assert len({id(r.env.known) for r in tasks})==8
    assert len({r.reward for r in tasks})>1
    before=model.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    result=update(policy,tasks,optimizer,dict(beta=0,clip=.2,epochs_per_group=epochs,max_grad_norm=1))
    assert result['optimizer_steps']==epochs
    assert result['zero_variance_group']==0
    assert not torch.equal(before,model.weight.detach())
    assert torch.isfinite(model.weight).all()


def test_slot_grammar_and_full_task_termination():
    policy=ConstrainedPolicy(tiny_model(),ByteTokenizer(),context_limit=50000)
    tasks=group(policy,fixture(),k=2,size=2,budget=2)
    for r in tasks:
        assert not r.env.failure and r.env.phase=='terminal'
        assert len(r.env.submission['paths'])==2
        assert r.env.moves<=2*(len(fixture().graph)-1)
        assert all(not p or tuple(p) in r.env.archive for p in r.env.submission['paths'])
