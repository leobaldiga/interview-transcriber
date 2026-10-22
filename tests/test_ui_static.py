"""Static UI consistency: every element id app.js looks up exists in index.html.

Catches the classic break where a template rename or refactor orphans a
getElementById call. No browser, no GPU — pure string checks.
"""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
APP_JS = (BASE / "static" / "js" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (BASE / "templates" / "index.html").read_text(encoding="utf-8")


def test_all_getelementby_id_targets_exist_in_index_html():
    ids = set(re.findall(r"getElementById\(\s*['\"]([\w-]+)['\"]\s*\)", APP_JS))
    assert ids, "no getElementById calls found — regex drift?"
    missing = sorted(i for i in ids if f'id="{i}"' not in INDEX_HTML)
    assert not missing, f"app.js references element ids missing from index.html: {missing}"


def test_compute_panel_elements_wired():
    for element_id in (
        "computeTranscriptionDevice",
        "computeDiarizationDevice",
        "computeWhisperPrecision",
        "computeSaveBtn",
        "computeTestBtn",
        "computeTestResult",
        "advTranscriptionDevice",
        "advDiarizationDevice",
    ):
        assert f'id="{element_id}"' in INDEX_HTML, element_id


def test_compute_endpoints_referenced_by_js_exist_in_app():
    for endpoint in (
        "/api/compute/devices",
        "/api/compute/settings",
        "/api/compute/tests",
        "/api/vram",
    ):
        assert endpoint in APP_JS, f"app.js no longer references {endpoint}"
