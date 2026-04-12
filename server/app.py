# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FlowState RL — Server entry point for multi-mode deployment.

This module is referenced by openenv.yaml as:
    app: server.app:app

It exposes the `app` FastAPI object that OpenEnv uses when running
the environment as an HTTP/WebSocket server.
"""

import sys
import os

# Ensure the parent directory (/app/env) is on the path so that
# `env` and `models` can be imported as top-level modules,
# matching how the Dockerfile sets PYTHONPATH="/app/env".
_env_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _env_root not in sys.path:
    sys.path.insert(0, _env_root)

from openenv_core import create_web_interface_app  # noqa: E402
from env import FlowStateEnv                        # noqa: E402
from models import BlockAction, BlockObservation    # noqa: E402

# Create the FastAPI application using the official OpenEnv factory.
# This wires up /reset, /step, /state, /health, and the web UI.
app = create_web_interface_app(FlowStateEnv, BlockAction, BlockObservation)


def main() -> None:
    """Run the server directly (useful for local testing)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
