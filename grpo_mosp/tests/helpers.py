"""Deterministic graphs and tiny byte-token policy for offline CPU checks."""
import networkx as nx
from ..dataset import Instance


def fixture():
    graph = nx.DiGraph()
    # A and B are distinct equal-cost Pareto routes. C is globally dominated.
    for u,v,c in [('s','a',(1,4)),('a','t',(1,1)),('s','b',(1,4)),('b','t',(1,1)),
                  ('s','c',(4,4)),('c','t',(1,2)),('s','z',(3,3))]:
        graph.add_edge(u,v,water=c[0],time=c[1])
    return Instance(graph,'s','t',('water','time'),{('s','a','t'),('s','b','t')},0)


class ByteTokenizer:
    def encode(self,text,add_special_tokens=False):
        return list(text.encode())
    def decode(self,tokens,skip_special_tokens=False):
        return bytes(tokens).decode()
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True,**kwargs):
        return self.encode(''.join(m['role']+':'+m['content']+'\n' for m in messages)+'assistant:')


def tiny_model():
    import torch
    from types import SimpleNamespace
    from contextlib import nullcontext
    class Tiny(torch.nn.Module):
        # Every prediction is trainable; no external model downloads are needed.
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(256))
        @property
        def device(self):
            return self.weight.device
        def forward(self,input_ids,use_cache=False,logits_to_keep=0):
            return SimpleNamespace(logits=self.weight.expand(1,min(logits_to_keep or input_ids.shape[1], input_ids.shape[1]),256))
        def disable_adapter(self):
            return nullcontext()
    return Tiny()
