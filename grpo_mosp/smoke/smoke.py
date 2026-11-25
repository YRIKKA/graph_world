"""CPU environment/evaluator smoke; requires only requirements-cpu.txt."""
import json
from ..dataset import generate
from ..evaluate import evaluate


def main():
    instances=[]
    for seed in range(1000):
        try: instances.append(generate(seed,k=3))
        except ValueError: continue
        if len(instances)==3: break
    if len(instances)!=3: raise RuntimeError('Could not generate smoke graphs')
    report=evaluate(instances)
    assert report['summary']['full_information']['pareto_fraction']==1
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__': main()
