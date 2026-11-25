"""Outcome GRPO, per-rollout token averaging, identical constrained supports."""
import torch


def advantages(rewards, epsilon=1e-8):
    rewards = torch.as_tensor(rewards, dtype=torch.float32).detach()
    std = rewards.std(unbiased=False)
    return ((rewards-rewards.mean())/(std+epsilon)).detach(), bool(std==0)


def surrogate(current, old, reference, advantage, clip=0.2, beta=0.01):
    ratio = (current-old).exp()
    # Sampled k3 estimator: exp(log(ref)-log(current))-log(ref)+log(current)-1.
    # Nonnegative pointwise; off-policy reuse makes it an approximation to KL.
    delta = reference-current
    kl = delta.exp()-delta-1
    return -(torch.minimum(ratio*advantage, ratio.clamp(1-clip,1+clip)*advantage)-beta*kl)


def update(policy, rollouts, optimizer, config):
    adv, zero = advantages([r.reward for r in rollouts])
    if zero and config['beta'] == 0:
        # Skip the optimizer too: Adam momentum must not move parameters when
        # this group has neither a reward signal nor a KL objective.
        return dict(loss=0.0, zero_variance_group=1, mean_reward=rollouts[0].reward,optimizer_steps=0,
                    generated_lengths=[sum(len(d.tokens) for d in r.decisions) for r in rollouts])
    # Reference = frozen base checkpoint via disabled LoRA adapters. Cache once.
    if config['beta']:
        with policy.model.disable_adapter(), torch.no_grad():
            for r in rollouts:
                for d in r.decisions:
                    d.reference_logps = policy.replay(d).cpu().tolist()
    total_loss = 0.0
    for _ in range(config['epochs_per_group']):
        optimizer.zero_grad(set_to_none=True)
        for i,r in enumerate(rollouts):
            length = sum(len(d.tokens) for d in r.decisions)
            for d in r.decisions:
                current = policy.replay(d)
                old = torch.tensor(d.old_logps, device=current.device).detach()
                reference = torch.tensor(d.reference_logps, device=current.device) if config['beta'] else current.detach()
                loss = surrogate(current,old,reference,adv[i].to(current.device),config['clip'],config['beta']).sum()/length/len(rollouts)
                loss.backward()  # Accumulate one action at a time to bound GPU memory.
                total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in policy.model.parameters() if p.requires_grad],config['max_grad_norm'])
        optimizer.step()
    return dict(loss=total_loss/config['epochs_per_group'], zero_variance_group=int(zero),optimizer_steps=config['epochs_per_group'],
                mean_reward=sum(r.reward for r in rollouts)/len(rollouts),
                generated_lengths=[sum(len(d.tokens) for d in r.decisions) for r in rollouts])
