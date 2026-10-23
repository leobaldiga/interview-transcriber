"""Test configuration — must run before app.py is imported by any test module.

app.py resolves its storage locations (uploads/, outputs/, compute settings)
from environment variables at import time. This conftest redirects all of
them to a throwaway directory so the test suite can never read or rewrite
real job data, and never touches a real compute configuration.
"""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="interview-transcriber-test-"))
os.environ["UPLOADS_DIR"] = str(_TEST_ROOT / "uploads")
os.environ["OUTPUTS_DIR"] = str(_TEST_ROOT / "outputs")
os.environ["COMPUTE_SETTINGS_PATH"] = str(_TEST_ROOT / "config" / "compute.json")
# Deterministic, non-secret token so app import never prompts/reads real files.
os.environ.setdefault("HF_TOKEN", "test-token-for-suite")
