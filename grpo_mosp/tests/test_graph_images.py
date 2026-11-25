"""Verify the requested plot_graph implementation sees exact saved edge weights."""
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from ..graph_images import export_graph_image
from jm_test.graph_world.generate_graph_world import GraphGenerator
from ..run_artifacts import artifact_environment, new_run_directory, validate_run_directory
from .helpers import fixture


def test_plot_graph_receives_exact_saved_topology_and_costs(tmp_path, monkeypatch):
    item = fixture()
    before = item.record()
    calls = []
    original = GraphGenerator.plot_graph
    def spy(self, graph, filename, **options):
        calls.append((deepcopy(graph), options, self.edge_costs))
        return original(self, graph, filename, **options)
    monkeypatch.setattr(GraphGenerator, 'plot_graph', spy)
    image = export_graph_image(item, tmp_path/'images'/'ground-truth.png')
    assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    graph, options, enabled = calls[0]
    assert list(graph.edges(data=True)) == list(item.graph.edges(data=True))
    assert options['edge_cost'] == ['water','time'] and enabled
    assert options['transition_probs'] is False
    assert graph.nodes[item.source]['role'] == 'source'
    assert graph.nodes[item.target]['role'] == 'target'
    assert item.record() == before


def test_naming_uses_client_timezone_and_cache_environment_is_scoped(tmp_path):
    now = datetime(2026,11,3,17,30,15,123456,tzinfo=ZoneInfo('UTC'))
    output = new_run_directory(tmp_path/'full_runs','Qwen/Qwen2.5-1.5B-Instruct',now)
    assert output.name == 'run_20261103123015'
    assert validate_run_directory(output,'Qwen/Qwen2.5-1.5B-Instruct') == output
    output.mkdir(parents=True)
    prior = os.environ.get('HF_HOME')
    with artifact_environment(output) as paths:
        assert Path(os.environ['HF_HOME']).is_relative_to(output)
        assert Path(tempfile.gettempdir()).is_relative_to(output)
        print('run-local console log')
    assert os.environ.get('HF_HOME') == prior
    assert 'run-local console log' in (output/'logs'/'launcher.log').read_text()


def test_child_process_log_and_temporary_files_stay_in_run(tmp_path):
    import sys
    from ..run_mosp import execute
    output = tmp_path/'full_runs'/'run_20261103123015'
    output.mkdir(parents=True)
    with artifact_environment(output):
        execute([sys.executable, '-c',
            "import os,tempfile; print(os.environ['HF_HOME']); print(tempfile.gettempdir()); print('stage output')"],
            output/'logs'/'test-stage.log')
    log = (output/'logs'/'test-stage.log').read_text()
    assert str(output/'cache'/'huggingface') in log and str(output/'tmp') in log
    assert 'stage output' in (output/'logs'/'launcher.log').read_text()


def test_artifact_context_does_not_report_an_exception_twice(tmp_path,capsys):
    from ..run_artifacts import artifact_environment
    import pytest
    with pytest.raises(ValueError,match='failure owned by CLI'):
        with artifact_environment(tmp_path/'run'):
            raise ValueError('failure owned by CLI')
    assert capsys.readouterr().err == ''


def test_whole_second_collisions_and_legacy_reuse(tmp_path):
    import pytest
    root = tmp_path/'full_runs'
    now = datetime(2026,11,3,17,30,15,123456,tzinfo=ZoneInfo('UTC'))
    output = new_run_directory(root,now=now)
    output.mkdir(parents=True)
    # Two launches in the same second must never silently share artifacts.
    with pytest.raises(FileExistsError,match='timestamp collision'):
        new_run_directory(root,now=now.replace(microsecond=999999))
    legacy = root/'run_20261103123015123456'
    with pytest.raises(ValueError,match='YYYYMMDDHHMMSS'):
        validate_run_directory(legacy)
    legacy.mkdir()
    assert validate_run_directory(legacy)==legacy.resolve()
    with pytest.raises(ValueError):
        validate_run_directory(root/'run_20261303123015')


def test_moo_prefixed_run_preserves_timestamp_and_validation(tmp_path):
    import pytest
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from ..figures import figure_root

    now = datetime(2026, 11, 13, 12, 30, 15, tzinfo=ZoneInfo('America/New_York'))
    output = new_run_directory(tmp_path/'full_runs', now=now, prefix='moo_rewards_run_')
    assert output.name == 'moo_rewards_run_20261113123015'
    assert validate_run_directory(output) == output
    assert figure_root({}, output/'Model_trained/moo_rewards/valid/runs/seed-17') == output/'figures'
    with pytest.raises(ValueError):
        validate_run_directory(tmp_path/'full_runs/moo_rewards_run_20261313123015')
