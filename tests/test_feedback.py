"""Compatibility collection entry point for the split feedback test suite.

The historical command ``pytest tests/test_feedback.py`` remains valid while
the actual tests live under ``tests/test_feedback/`` by responsibility.
"""

import sys

# The normal repository collection already discovers ``feedback_suite``.  Only
# expose the classes here when the historical file path is explicitly passed,
# so the compatibility entry point does not run every test twice.
_explicit_compat_run = any(
    arg in {"test_feedback.py", "tests/test_feedback.py"}
    or arg.endswith("/tests/test_feedback.py")
    for arg in sys.argv[1:]
)

if _explicit_compat_run:
    from tests.feedback_suite.test_audio import *  # noqa: F401,F403
    from tests.feedback_suite.test_core import *  # noqa: F401,F403
    from tests.feedback_suite.test_integration import *  # noqa: F401,F403
    from tests.feedback_suite.test_quality import *  # noqa: F401,F403
else:
    __test__ = False
