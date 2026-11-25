"""Wall-clock component timing for logs, with append-only invocation summaries."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def duration(seconds):
    """Keep days/hours visible for long single-GPU experiments."""
    milliseconds = round(max(0.,seconds)*1000)
    seconds,ms = divmod(milliseconds,1000)
    minutes,seconds = divmod(seconds,60)
    hours,minutes = divmod(minutes,60)
    days,hours = divmod(hours,24)
    return f'{days}d {hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}'


@contextmanager
def timed_component(label, records=None, announce=True):
    """Durations are inclusive: a child component is part of its parent stage."""
    record = dict(component=label,status='completed')
    start = time.perf_counter()
    if announce:
        print(f'[Timing] START {label}',flush=True)
    try:
        yield record
    except BaseException as error:
        record['status'] = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        raise
    finally:
        record['seconds'] = time.perf_counter()-start
        if records is not None:
            records.append(record)
        if announce:
            print(f'[Timing] END {label} | {record["status"]} | '
                  f'{duration(record["seconds"])} ({record["seconds"]:.3f} seconds)',flush=True)


class PipelineTiming:
    def __init__(self, started_at=None):
        current = time.time()
        try:
            value = float(started_at)
        except (TypeError,ValueError):
            value = current
        self.started_at = value if math.isfinite(value) and 0<value<=current else current
        # Use a monotonic timer after initialization; only the shell-startup
        # offset uses wall-clock time, because it crosses the exec boundary.
        self.startup_seconds = current-self.started_at
        self.monotonic_start = time.perf_counter()
        self.records = []

    def component(self,label,announce=True):
        return timed_component(label,self.records,announce)

    def finish(self,output,model,mode,status):
        from .reporting import table
        total = self.startup_seconds+time.perf_counter()-self.monotonic_start
        record = dict(model=model,mode=mode,status=status,started_at_unix=self.started_at,
            started_at_utc=datetime.fromtimestamp(self.started_at,timezone.utc).isoformat(),
            finished_at_unix=time.time(),total_seconds=total,components=self.records,
            scope='this launcher invocation, including shell/Conda startup when run_mosp.sh is used; reused stages only include verification time')
        path = Path(output)/'pipeline_timings.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        table(f'{model} | pipeline component times | mode={mode} | {status}',
              ['Component','Status','Elapsed','Seconds'],
              [[r['component'],r['status'],duration(r['seconds']),f"{r['seconds']:.3f}"] for r in self.records]+
              [['Launcher/startup/other overhead','',duration(max(0.,total-sum(r['seconds'] for r in self.records))),
                f"{max(0.,total-sum(r['seconds'] for r in self.records)):.3f}"]])
        # Keep this as the final console/log line, after the table and exports.
        total = self.startup_seconds+time.perf_counter()-self.monotonic_start
        print(f'TOTAL PIPELINE ELAPSED TIME: {duration(total)} ({total:.3f} seconds) | '
              f'status={status} | mode={mode}',flush=True)
