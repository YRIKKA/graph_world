"""Timing must include startup, respect reused work, and end the master log."""
import json
import time

import pytest

from ..pipeline_timing import duration, timed_component, PipelineTiming
from .. import run_mosp as runner


def test_duration_supports_long_experiments():
    assert duration(90061.234)=='1d 01:01:01.234'
    assert duration(.9999)=='0d 00:00:01.000'


def test_component_failures_and_reuse_are_recorded(capsys):
    records=[]
    with timed_component('reuse stage',records) as measurement:
        measurement['status']='reused'
    with pytest.raises(ValueError):
        with timed_component('failed stage',records):
            raise ValueError('failure')
    assert [r['status'] for r in records]==['reused','failed']
    assert all(r['seconds']>=0 for r in records)
    assert '[Timing] END reuse stage | reused' in capsys.readouterr().out


def test_launch_total_includes_startup_and_is_final_line(tmp_path,monkeypatch,capsys):
    root=tmp_path/'full_runs/run_20260101120000'
    args=runner.parser().parse_args(['--mode','prepare','--output',str(root)])
    monkeypatch.setenv('MOSP_PIPELINE_STARTED_AT',str(time.time()-20))
    monkeypatch.setattr(runner,'dependencies',lambda *args:None)
    monkeypatch.setattr(runner,'make_data',lambda *args:None)
    runner.launch(args,vars(args))
    text=(root/'log.txt').read_text()
    assert text.splitlines()[-1].startswith('TOTAL PIPELINE ELAPSED TIME:')
    assert 'status=completed' in text.splitlines()[-1]
    record=json.loads((root/'pipeline_timings.jsonl').read_text())
    assert record['total_seconds']>=20
    assert record['status']=='completed' and record['mode']=='prepare'
    assert [r['component'] for r in record['components']]==[
        'Resolve settings and construct plan','Dependency checks',
        'Training/validation graph data and weighted images','Write model registry and reproducible configurations']
    assert text==capsys.readouterr().out


def test_failed_launch_also_finishes_with_elapsed_time(tmp_path,monkeypatch):
    root=tmp_path/'full_runs/run_20260101120000'
    args=runner.parser().parse_args(['--mode','prepare','--output',str(root)])
    def fail(*args):raise RuntimeError('deliberate failure')
    monkeypatch.setattr(runner,'_launch_actual',fail)
    with pytest.raises(SystemExit) as raised:runner.launch(args,vars(args))
    assert raised.value.code==1
    last=(root/'log.txt').read_text().splitlines()[-1]
    assert last.startswith('TOTAL PIPELINE ELAPSED TIME:') and 'status=failed' in last
    assert json.loads((root/'pipeline_timings.jsonl').read_text())['status']=='failed'


def test_invocation_records_append_instead_of_overwriting(tmp_path,capsys):
    for mode in ('train','eval'):
        timing=PipelineTiming()
        with timing.component('verify saved artifacts') as record:record['status']='reused'
        timing.finish(tmp_path,'Synthetic/Test',mode,'completed')
    rows=[json.loads(line) for line in (tmp_path/'pipeline_timings.jsonl').read_text().splitlines()]
    assert [r['mode'] for r in rows]==['train','eval']
    assert all(r['components'][0]['status']=='reused' for r in rows)
