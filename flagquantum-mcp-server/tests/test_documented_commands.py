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

# What a relative path this repository owes anyone looks like. The wheel block
# names its interpreter through a shell variable (``"$WHEELCHECK/bin/python"``)
# and names the packaged module inside a line of Python, and both contain a
# slash without being paths a contributor could run from here. Anything a shell
# would expand, quote or parse is not a token this check can resolve, so it is
# not one the check may claim is missing.
PATH_LIKE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./-]*$")

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
    prose, not something anyone is told to run. A token that is not shaped like
    a relative path — quoted, or carrying a shell expansion, or a fragment of
    the Python the block passes to ``-c`` — is not collected either, because
    ``(PACKAGE_ROOT / token)`` cannot resolve it and reporting it as missing
    would be a false alarm about a command that runs.
    """
    tokens: list[str] = []
    for block in _verification_blocks():
        for line in block.splitlines():
            code = line.split("#", 1)[0]
            for token in code.split():
                if token.startswith("/") or "*" in token or "://" in token:
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
