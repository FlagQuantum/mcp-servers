"""Support ``python -m flagquantum_mcp_server``.

Equivalent to the ``flagquantum-mcp-server`` console script, and useful in
environments where the venv's ``bin`` directory is not on ``PATH`` — a CI job
or an editor-spawned process, for instance.
"""

from __future__ import annotations

from flagquantum_mcp_server import main

if __name__ == "__main__":
    main()
