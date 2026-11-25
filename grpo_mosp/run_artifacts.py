"""Timestamped run identities and run-local caches/logs for run_mosp.sh."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import tempfile
from zoneinfo import ZoneInfo

BASELINE_PRESET_DIR = Path(__file__).resolve().parent/'experiment_eval_baselines'/'configs'
RUN_TIMEZONE = 'America/New_York'
TIMESTAMP_FORMAT = '%Y%m%d%H%M%S'
RUN_PREFIXES = ('run_', 'moo_rewards_run_')


def model_slug(model):
    # A Hub namespace contains '/', which must not create nested run directories.
    name = str(model).rstrip('/').rsplit('/', 1)[-1] or 'model'
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def new_run_directory(root, model=None, now=None, *, prefix='run_'):
    root = Path(root).expanduser().resolve()
    if root.name != 'full_runs':
        raise ValueError('--runs-root must name a directory called full_runs')
    if prefix not in RUN_PREFIXES:
        raise ValueError(f'Unsupported run prefix: {prefix}')
    now = now or datetime.now(ZoneInfo(RUN_TIMEZONE))
    timestamp = now.astimezone(ZoneInfo(RUN_TIMEZONE)).strftime(TIMESTAMP_FORMAT)
    path = root / f'{prefix}{timestamp}'
    if path.exists():
        raise FileExistsError(f'Run timestamp collision: {path}; retry with a fresh timestamp')
    return path


def validate_run_directory(path, model=None):
    path = Path(path).expanduser().resolve()
    prefix = next((value for value in RUN_PREFIXES if path.name.startswith(value)), None)
    if path.parent.name != 'full_runs' or prefix is None:
        raise ValueError('Run outputs must be full_runs/run_YearMonthDayTime or full_runs/moo_rewards_run_YearMonthDayTime; use --runs-root to choose its parent location')
    timestamp = path.name[len(prefix):]
    # Existing microsecond-named runs remain usable for evaluation/reuse. New
    # directories, including explicit --output paths, use whole seconds only.
    legacy = len(timestamp)==20 and path.is_dir()
    if (len(timestamp)!=14 and not legacy) or not timestamp.isdigit():
        raise ValueError('Run timestamp must be YYYYMMDDHHMMSS')
    datetime.strptime(timestamp, TIMESTAMP_FORMAT+'%f' if legacy else TIMESTAMP_FORMAT)
    return path


class Tee:
    """Keep terminal progress while writing the same output into the run folder."""
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()
        return len(value)
    def __getattr__(self,name):
        return getattr(self.terminal,name)
    def isatty(self):
        return self.terminal.isatty()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


@contextmanager
def console_log(path, extra_paths=()):
    """Append Python output to log.txt, avoiding duplicate parent/child logging.

    The launcher also captures the child's OS stdout/stderr pipe, including
    native-library output, in this same file. Its child inherits a marker so it
    does not write that file a second time. Distinct nested checkpoint logs are
    still populated, and aliases to the same file are opened only once.
    """
    paths = list(dict.fromkeys(Path(p).resolve() for p in (path,*extra_paths)))
    managed = os.environ.get('MOSP_CONSOLE_LOG_PATH')
    if managed:
        paths = [p for p in paths if p != Path(managed).resolve()]
    with ExitStack() as stack:
        streams = []
        for target in paths:
            target.parent.mkdir(parents=True,exist_ok=True)
            streams.append(stack.enter_context(target.open('a',buffering=1)))
        stdout,stderr = sys.stdout,sys.stderr
        for stream in streams:
            stdout,stderr = Tee(stdout,stream),Tee(stderr,stream)
        with redirect_stdout(stdout),redirect_stderr(stderr):
            yield


@contextmanager
def artifact_environment(output, log_directory=None):
    """Redirect model downloads, plotting/compilation caches and temporary files.

    Set this BEFORE importing Torch/Transformers/Matplotlib. Child processes
    inherit the same paths, and all independent training runs share this cache.
    """
    output = Path(output)
    paths = {
        'HF_HOME': output/'cache'/'huggingface',
        'HF_HUB_CACHE': output/'cache'/'huggingface'/'hub',
        'HUGGINGFACE_HUB_CACHE': output/'cache'/'huggingface'/'hub',
        'HF_DATASETS_CACHE': output/'cache'/'datasets',
        'TRANSFORMERS_CACHE': output/'cache'/'huggingface'/'hub',
        'TORCH_HOME': output/'cache'/'torch',
        'XDG_CACHE_HOME': output/'cache'/'xdg',
        'MPLCONFIGDIR': output/'cache'/'matplotlib',
        'TRITON_CACHE_DIR': output/'cache'/'triton',
        'UNSLOTH_COMPILE_LOCATION': output/'cache'/'unsloth_compiled_cache',
        'TORCHINDUCTOR_CACHE_DIR': output/'cache'/'torchinductor',
        'CUDA_CACHE_PATH': output/'cache'/'cuda',
        'PYTHONPYCACHEPREFIX': output/'cache'/'pycache',
        'TMPDIR': output/'tmp', 'TMP': output/'tmp', 'TEMP': output/'tmp',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    values = {key: str(path) for key, path in paths.items()}
    values['PYTHONDONTWRITEBYTECODE'] = '1'
    previous = {key: os.environ.get(key) for key in values}
    old_tempdir = tempfile.tempdir
    os.environ.update(values)
    tempfile.tempdir = None
    logs = Path(log_directory) if log_directory is not None else output/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with console_log(output/'log.txt',extra_paths=[logs/'launcher.log']):
            yield values
    finally:
        tempfile.tempdir = old_tempdir
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
