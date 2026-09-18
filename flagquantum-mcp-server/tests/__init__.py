"""Make ``tests`` importable as a package.

Two modules here are imported by name rather than by pytest's own collection.
``surface.py`` holds the server's declared surface and is read by the CI job
that smoke-tests the built wheel — an environment with no development
dependencies — so it imports nothing. ``conftest.py`` holds the fixtures and
the shared circuit constants, and tests import it as ``tests.conftest``.

Neither import resolved before this file existed, and the failure was invisible
from the command this repository documents in one place and not another:
``python -m pytest`` inserts the working directory into ``sys.path``, so the
import succeeded locally, while ``pytest`` — the console script CI runs —
inserts only the test file's directory. Two invocations, two different import
paths, and the one that passed was the one that was wrong.
"""
