"""Exercise the vLLM socket-directory lifecycle without a model server."""

from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, Mock

import pytest

from ..experiment_arc3.duck import vllm_backend


@pytest.mark.parametrize("ready", [True, False], ids=["ready", "startup-failure"])
@pytest.mark.parametrize("override", [False, True], ids=["private", "caller-managed"])
def test_serve_vllm_socket_directory(tmp_path, monkeypatch, ready, override):
    run_tmp = str(tmp_path / ("long-run-directory" * 8) / "tmp")
    monkeypatch.setenv("TMPDIR", run_tmp)
    monkeypatch.delenv("VLLM_RPC_BASE_PATH", raising=False)
    caller_directory = tmp_path / "caller-ipc"
    if override:
        caller_directory.mkdir()
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(caller_directory))

    bound_socket = MagicMock()
    bound_socket.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 8123)
    monkeypatch.setattr(vllm_backend.socket, "socket", lambda: bound_socket)
    monkeypatch.setattr(vllm_backend.os, "killpg", Mock())
    monkeypatch.setattr(
        vllm_backend, "urlopen", lambda *args, **kwargs: BytesIO(b'{"data":[{"id":"base"}]}')
    )

    process = Mock(pid=10)
    process.poll.return_value = None if ready else 1
    socket_directory = None

    @contextmanager
    def start_server(command, *, env, **kwargs):
        nonlocal socket_directory
        socket_directory = Path(env["VLLM_RPC_BASE_PATH"])
        assert env["TMPDIR"] == run_tmp
        assert socket_directory.is_dir()
        if override:
            assert socket_directory == caller_directory
        else:
            assert socket_directory.parent == Path("/tmp")
            assert socket_directory.stat().st_mode & 0o777 == 0o700
            assert len(os.fsencode(socket_directory)) + 37 <= 107
        try:
            yield process
        finally:
            assert socket_directory.is_dir()

    monkeypatch.setattr(vllm_backend.subprocess, "Popen", start_server)
    launcher = vllm_backend.serve_vllm(
        sys.executable, tmp_path, "base", tmp_path / "server.log",
        context_limit=32768, concurrent_jobs=1,
    )
    if ready:
        with launcher as endpoint:
            assert endpoint == "http://127.0.0.1:8123/v1"
    else:
        with pytest.raises(RuntimeError, match="vLLM server failed to start"), launcher:
            pytest.fail("The subprocess exited before readiness")

    process.wait.assert_called_once_with(timeout=15)
    assert socket_directory.exists() == override
    assert os.environ.get("VLLM_RPC_BASE_PATH") == (str(caller_directory) if override else None)
