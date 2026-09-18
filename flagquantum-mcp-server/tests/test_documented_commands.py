"""The commands AGENTS.md tells an agent to run must actually resolve.

The verification section is the one part of this repository a contributor is
told to follow rather than read, and it is the part nothing checked. It
contained `../../.venv/bin/ruff` — one directory too far up, so every command
in it named a binary that does not exist. The cost of that is not a confusing
error message; it is that the person following it either stops and fixes the
doc, or silently substitutes a working equivalent and reports the result. The
second is what happened here, and it is how two red commits reached `main` and
a release bump: the substituted command was `python -m pytest`, which resolves
imports CI cannot.

So the paths are parsed and checked. They are relative to the package
directory, because that is where the block tells you to `cd` first.

Skipped where there is no `.venv`: the block describes a developer checkout,
and CI installs the package directly rather than into a venv.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_ROOT = REPO_ROOT / "flagquantum-mcp-server"
INSTRUCTIONS = REPO_ROOT / "AGENTS.md"

# What a relative path this repository owes anyone looks like, once the quotes
# are off it. Stripping them first is the point: a path written as
# ``"../.venv/bin/ruff"`` is one a contributor could run, and skipping it for
# being quoted would be a silent hole in this check. What legitimately fails this
# shape is a token a shell would have to expand or that is not a path at all --
# ``"$WHEELCHECK/bin/python"``, ``dist/*.whl``, ``dist/:``.
PATH_LIKE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./-]*$")

# A heredoc body is the language it is passed to, not shell: tokens in it are
# Python, and no contributor is told to run them. Matching the terminator closes
# the body so its contents are never tokenized.
HEREDOC = re.compile(r"<<'?(\w+)'?\n.*?\n\1\n", re.DOTALL)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not (REPO_ROOT / ".venv").exists(),
        reason="AGENTS.md describes a developer checkout; CI installs the package directly",
    ),
]


def _verification_blocks() -> list[str]:
    """Return the fenced bash blocks under ``## Verification``."""
    text = INSTRUCTIONS.read_text(encoding="utf-8")
    section = text.split("## Verification", 1)[1]
    return re.findall(r"```bash\n(.*?)```", section, re.DOTALL)


def _path_tokens() -> list[str]:
    """Return every relative path the blocks name, in order.

    Comments are stripped first: an example path inside a trailing comment is
    prose, not something anyone is told to run. Heredoc bodies are stripped too,
    for the same reason — they are the language the block passes them to, not
    shell. What remains is shell, and a token in it that carries a slash is
    collected whether or not it is quoted, because a quoted path is still a path
    someone can run.

    What is *not* collected is a token the shell would have to expand
    (``$WHEELCHECK/bin/python``), one that is a glob rather than a path, and one
    that is not path-shaped at all. ``(PACKAGE_ROOT / token)`` cannot resolve
    those, and reporting them missing would be a false alarm about a command that
    runs.
    """
    tokens: list[str] = []
    for block in _verification_blocks():
        for line in HEREDOC.sub("", block).splitlines():
            code = line.split("#", 1)[0]
            for raw in code.split():
                token = raw.strip("\"'`")
                if "$" in token or "*" in token or "://" in token or token.startswith("/"):
                    continue
                if "/" in token and PATH_LIKE.match(token):
                    tokens.append(token)
    return tokens


def test_the_verification_section_has_commands_to_check() -> None:
    """Guards the parsing: a rewrite that hides the blocks must not pass."""
    assert _verification_blocks()
    assert len(_path_tokens()) >= 5


def test_every_path_the_instructions_name_exists() -> None:
    missing = [token for token in _path_tokens() if not (PACKAGE_ROOT / token).exists()]

    assert missing == [], (
        f"AGENTS.md tells a contributor to run paths that do not exist: {missing}. "
        "A command in that section that cannot be run is worse than a missing one: "
        "it invites substituting a working equivalent, which is a different check."
    )


def test_the_documented_test_command_is_the_console_script() -> None:
    """The spelling is load-bearing, so it is pinned rather than trusted.

    ``pytest`` and ``python -m pytest`` put different directories on sys.path,
    and the difference is a collection error CI sees and a green local run.
    Only the bash blocks are read: the prose around them names the wrong
    spelling on purpose, to explain why it is wrong.
    """
    for block in _verification_blocks():
        assert "pytest -m" in block
        assert "python -m pytest" not in block, (
            "a block must not offer a second spelling of the test command"
        )
