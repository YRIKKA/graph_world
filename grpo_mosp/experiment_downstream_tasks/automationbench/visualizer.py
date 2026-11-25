"""Run the pinned upstream visualizer on this host's loopback interface."""

import argparse
import importlib.util
import socketserver
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ready", type=Path)
    args = parser.parse_args()
    source = args.checkout / "visualizer/serve.py"
    spec = importlib.util.spec_from_file_location("upstream_visualizer", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    class LoopbackServer(socketserver.TCPServer):
        def __init__(self, address, handler):
            super().__init__(("127.0.0.1", address[1]), handler)

        def server_activate(self):
            super().server_activate()
            if args.ready:
                args.ready.touch()

    upstream.PORT = args.port
    upstream.socketserver = SimpleNamespace(TCPServer=LoopbackServer)
    upstream.webbrowser = SimpleNamespace(open=lambda _: False)
    upstream.main()


if __name__ == "__main__":
    main()
