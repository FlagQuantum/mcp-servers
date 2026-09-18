"""Make ``tests`` importable as a package.

``conftest`` holds the server's declared surface — the tools, resources and
prompts — so that a new tool has one place to register rather than three that
drift. Tests import it as ``tests.conftest``, which only works when this
directory is a package.

It was not, and the failure was invisible from the command this repository
documents in one place and not another: ``python -m pytest`` inserts the
working directory into ``sys.path``, so the import resolved locally, while
``pytest`` — the console script CI runs — inserts only the test file's
directory. Two invocations, two different import paths, and the one that passed
was the one that was wrong.
"""
