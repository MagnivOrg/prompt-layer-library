# This file need to be in the root of repo, so it is imported before tests
import pytest

# we need this to get assert diffs everywhere `tests.*`, it must execute before importing `tests`
pytest.register_assert_rewrite("tests")

from tests.fixtures import *  # noqa: F401, F403, E402
