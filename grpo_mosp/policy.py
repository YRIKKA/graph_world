"""Token-trie constrained sampling with replayable probabilities and chat history."""
from dataclasses import dataclass, field
import json
from collections.abc import Mapping
import torch
from .model_options import thinking_options


class ContextOverflow(RuntimeError):
    """An explicit resource-limit outcome, never an instruction to truncate."""
    def __init__(self,length,limit):
        self.length,self.limit = length,limit
        super().__init__(f'Context overflow: {length}>{limit}; use --history-format state_actions, reduce task size, or profile a larger --context-limit')


@dataclass
class Decision:
    context: list
    tokens: list = field(default_factory=list)
    allowed: list = field(default_factory=list)
    old_logps: list = field(default_factory=list)
    reference_logps: list = field(default_factory=list)


def masked_logps(logits, allowed, temperature=1.0):
    # Normalize over precisely the legal trie children, including forced tokens.
    ids = torch.tensor(allowed, device=logits.device)
    return torch.log_softmax(logits[ids].float()/temperature, dim=-1)


class ConstrainedPolicy:
    def __init__(self, model, tokenizer, context_limit=32768, temperature=1.0, enable_thinking=None, history_format='full', reasoning_effort='xhigh', preserve_thinking=False):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.model, self.tokenizer = model, tokenizer
        self.context_limit, self.temperature = context_limit, temperature
        self.enable_thinking = enable_thinking
        options = thinking_options(dict(reasoning_effort=reasoning_effort, preserve_thinking=preserve_thinking))
        self.reasoning_effort = options['reasoning_effort']
        self.preserve_thinking = options['preserve_thinking']
        if history_format not in ('full','delta','state_actions'):
            raise ValueError('history_format must be full, delta, or state_actions')
        self.history_format = history_format

    @property
    def device(self):
        # Offloaded embeddings can make model.device report CPU even though
        # language layers run on CUDA. The backend records the actual input device.
        return getattr(self.model,'_mosp_device',self.model.device)

    def prompt(self, messages):
        if self.history_format == 'state_actions':
            from .history import compact_state_actions
            messages = compact_state_actions(messages)
        elif self.history_format == 'delta':
            from .history import compact_history
            messages = compact_history(messages)
        # These are template inputs, never model.forward/generate arguments.
        # Keep identical settings for rollout collection and frozen evaluation.
        kwargs = thinking_options(vars(self))
        if self.enable_thinking is not None:
            kwargs['enable_thinking'] = self.enable_thinking
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
        # Transformers 5 can return BatchEncoding by default; older/patched
        # tokenizers return a flat list. Always count/feed actual input token IDs.
        if isinstance(tokens,Mapping):
            tokens = tokens['input_ids']
        if hasattr(tokens,'tolist'):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0],list):
            if len(tokens)!=1:
                raise ValueError('Expected one tokenized conversation')
            tokens = tokens[0]
        return list(tokens)

    def check_length(self, length):
        if length > self.context_limit:
            # Refuse the run: truncation would change the POMDP and likelihoods.
            raise ContextOverflow(length,self.context_limit)

    def check_token_budget(self):
        """Evaluation can override this hook; training has no extra token cap."""

    def record_token(self):
        """Evaluation accounting hook, called once per generated assistant token."""

    def sample(self, context, candidates):
        sequences = sorted({tuple(self.tokenizer.encode(s, add_special_tokens=False)) for s in candidates})
        if not sequences or any(not s for s in sequences):
            raise ValueError('Empty grammar')
        # Candidate strings here end at the same schema boundary; no stop token.
        # Reject prefix ambiguity instead of silently changing probabilities.
        if any(a != b and b[:len(a)] == a for a in sequences for b in sequences):
            raise ValueError('Grammar candidates must be prefix-free')
        self.check_length(len(context)+max(map(len,sequences)))
        decision = Decision(list(context))
        while tuple(decision.tokens) not in sequences:
            self.check_token_budget()
            prefix = tuple(decision.tokens)
            allowed = sorted({s[len(prefix)] for s in sequences if s[:len(prefix)]==prefix})
            with torch.no_grad():
                # Forced schema tokens have exactly probability one; avoid a forward.
                if len(allowed)==1:
                    token, logp = allowed[0], 0.0
                else:
                    ids = torch.tensor([context+decision.tokens], device=self.device)
                    logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0,-1]
                    lp = masked_logps(logits, allowed, self.temperature)
                    index = torch.multinomial(lp.exp(), 1).item()
                    token, logp = allowed[index], lp[index].item()
            decision.allowed.append(allowed)
            decision.tokens.append(token)
            self.record_token()
            decision.old_logps.append(logp)
        return decision

    def replay(self, decision):
        # Only assistant output positions are gathered. Environment tokens are
        # conditioning inputs and have no direct loss positions.
        self.check_length(len(decision.context)+len(decision.tokens))
        if all(len(a)==1 for a in decision.allowed):
            # Forced-only actions have identically zero log probabilities. Keep a
            # zero gradient connection so backward works without an LM forward.
            parameter = next((p for p in self.model.parameters() if p.requires_grad), None)
            if parameter is None or not torch.is_grad_enabled():
                return torch.zeros(len(decision.tokens), device=self.device)
            return (parameter.reshape(-1)[0]*0).expand(len(decision.tokens))
        ids = torch.tensor([decision.context+decision.tokens], device=self.device)
        # Project only assistant prediction positions onto the model's
        # vocabulary; projecting the entire history would waste many GB of VRAM.
        logits = self.model(input_ids=ids, use_cache=False,
                            logits_to_keep=len(decision.tokens)+1).logits[0]
        start = 0
        return torch.stack([masked_logps(logits[start+i], allowed, self.temperature)[allowed.index(token)]
                            for i,(token,allowed) in enumerate(zip(decision.tokens,decision.allowed))])

    def action(self, messages, env):
        # Retain completed chunks if a later final-submission slot hits a limit.
        self.pending_decisions = []
        context = self.prompt(messages)
        if env.phase == 'traversal':
            d = self.sample(context, [json.dumps(a,separators=(',',':')) for a in env.choices()])
            self.pending_decisions.append(d)
            return self.tokenizer.decode(d.tokens, skip_special_tokens=False), self.pending_decisions
        # Factor the submission grammar by slot. This represents every ordered
        # k-slot combination without constructing (archive_size+1)**k strings.
        decisions, output = self.pending_decisions, []
        options = [[], *[list(p) for p in sorted(env.archive)]]
        for slot in range(env.k):
            prefix = '{"paths":[' if slot==0 else ','
            suffix = ']}' if slot==env.k-1 else ''
            candidates = [prefix+json.dumps(path,separators=(',',':'))+suffix for path in options]
            d = self.sample(context+output, candidates)
            output.extend(d.tokens)
            decisions.append(d)
        return self.tokenizer.decode(output, skip_special_tokens=False), decisions


def load_policy(config, adapter=None, *, training=True):
    """Training and evaluation share the exact same frozen-base loader."""
    from .model_backends import load_model
    from .model_defaults import initial_checkpoint
    spec = initial_checkpoint(config)
    if adapter:
        spec['adapter'] = adapter
    model, tokenizer = load_model(config, spec, training=training and adapter is None)
    return ConstrainedPolicy(model,tokenizer,config['context_limit'],config['temperature'],
        enable_thinking=getattr(model,'_mosp_loading',{}).get('enable_thinking'),
        history_format=config.get('history_format','full'), **thinking_options(config))
