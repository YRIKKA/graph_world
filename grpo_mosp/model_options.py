"""Shared Qwen chat-template controls for every model-facing entry point."""
import argparse

REASONING_EFFORTS = ('xhigh', 'medium', 'low')


def thinking_options(config):
    """Resolve missing fields without changing old checkpoint loading identities."""
    effort = config.get('reasoning_effort', 'xhigh')
    preserve = config.get('preserve_thinking', False)
    if effort not in REASONING_EFFORTS:
        raise ValueError(f'reasoning_effort must be one of {REASONING_EFFORTS}')
    # Do not accept truthy strings: bool("false") would silently enable retention.
    if not isinstance(preserve, bool):
        raise ValueError('preserve_thinking must be a boolean')
    return dict(reasoning_effort=effort, preserve_thinking=preserve)


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Use true or false')
    return value.lower() == 'true'


def add_thinking_arguments(parser):
    # Suppression lets CLI overrides win over JSON without replacing saved values
    # when a flag is omitted. Both spellings are accepted by all entry points.
    parser.add_argument('--reasoning-effort', '--reasoning_effort', choices=REASONING_EFFORTS,
                        default=argparse.SUPPRESS, help='Qwen template reasoning effort [default: xhigh]')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preserve-thinking', '--preserve_thinking', type=boolean, nargs='?', const=True,
                       default=argparse.SUPPRESS, help='Retain previous assistant thinking in the template [default: false]')
    group.add_argument('--no-preserve-thinking', dest='preserve_thinking', action='store_false',
                       default=argparse.SUPPRESS, help='Alias for --preserve-thinking false')


def apply_thinking_arguments(config, args):
    """Merge only supplied flags, then persist the resolved effective defaults."""
    for key in ('reasoning_effort', 'preserve_thinking'):
        if hasattr(args, key):
            config[key] = getattr(args, key)
    config.update(thinking_options(config))
    return config
