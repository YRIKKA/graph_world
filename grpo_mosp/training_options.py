"""Shared training schedule defaults; validation controls never reference test data."""
import argparse
import math
from .model_options import boolean

TRAINING_DEFAULTS = dict(steps=30, epochs_per_group=1, early_stopping=True,
    early_stopping_interval=5, early_stopping_patience=3, early_stopping_min_delta=0.,
    early_stopping_graphs=8, early_stopping_repetitions=1, early_stopping_seed=314159)


def training_options(config):
    result = {key:config.get(key,value) for key,value in TRAINING_DEFAULTS.items()}
    if not isinstance(result['early_stopping'],bool):
        raise ValueError('early_stopping must be a boolean')
    for key,value in result.items():
        if key in ('early_stopping','early_stopping_min_delta'):
            continue
        if type(value) is not int or value < (0 if key=='early_stopping_seed' else 1):
            raise ValueError(f'{key} must be a positive integer (seed may be zero)')
    delta = result['early_stopping_min_delta']
    if not isinstance(delta,(int,float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError('early_stopping_min_delta must be finite and >= 0')
    return result


def add_training_arguments(parser, schedule=True):
    """Omitted arguments preserve JSON values; defaults are resolved after merging."""
    if schedule:
        for key in ('steps','epochs_per_group'):
            parser.add_argument('--'+key.replace('_','-'),type=int,default=argparse.SUPPRESS,
                                help=f'[default: {TRAINING_DEFAULTS[key]}]')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--early-stopping',type=boolean,nargs='?',const=True,default=argparse.SUPPRESS,
                       help='Export the best validation adapter and stop on a plateau [default: true]')
    group.add_argument('--no-early-stopping',dest='early_stopping',action='store_false',default=argparse.SUPPRESS)
    for key in ('early_stopping_interval','early_stopping_patience','early_stopping_graphs',
                'early_stopping_repetitions','early_stopping_seed','early_stopping_min_delta'):
        parser.add_argument('--'+key.replace('_','-'),type=float if key.endswith('min_delta') else int,
                            default=argparse.SUPPRESS,help=f'[default: {TRAINING_DEFAULTS[key]}]')


def apply_training_arguments(config,args=None):
    for key in TRAINING_DEFAULTS:
        if hasattr(args,key):
            config[key] = getattr(args,key)
    config.update(training_options(config))
    return config
