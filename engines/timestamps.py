"""Timestamp parsing for heterogeneous whisper.cpp builds.

Current upstream `whisper-server` returns `verbose_json` timestamps as floats in
seconds (`t0 * 0.01`). Older builds and the CLI `-oj` writer instead emit
`offsets.from/to` in integer milliseconds and `timestamps.from/to` as
`"00:00:12,340"` strings. Rather than sniffing units from magnitude — which
silently turns a 90-minute interview into a 90-millisecond one — the unit is
always passed explicitly by the caller that knows which field it read.
"""

from __future__ import annotations

import re

_CLOCK_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<frac>\d{1,9}))?$"
)


def parse_timestamp(value, unit: str = "s") -> float:
    """Return seconds as a float.

    `unit` applies to bare numbers only: "s" for seconds, "ms" for milliseconds.
    Clock-formatted strings ("00:00:12,340", "1:02:03.5", "02:07") always carry
    their own unit and ignore `unit`. Bare numeric strings ("12.34") are treated
    as `unit`.
    """
    if unit not in ("s", "ms"):
        raise ValueError(f"unit must be 's' or 'ms', got {unit!r}")

    if isinstance(value, bool) or value is None:
        raise ValueError(f"not a timestamp: {value!r}")

    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds / 1000.0 if unit == "ms" else seconds

    if not isinstance(value, str):
        raise ValueError(f"not a timestamp: {value!r}")

    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")

    negative = text.startswith("-")
    if negative:
        text = text[1:]

    match = _CLOCK_RE.match(text)
    if match:
        hours = int(match.group("h") or 0)
        minutes = int(match.group("m"))
        secs = int(match.group("s"))
        frac = match.group("frac") or ""
        fraction = float(f"0.{frac}") if frac else 0.0
        seconds = hours * 3600 + minutes * 60 + secs + fraction
    else:
        try:
            seconds = float(text)
        except ValueError as exc:
            raise ValueError(f"unrecognised timestamp format: {value!r}") from exc
        if unit == "ms":
            seconds /= 1000.0

    return -seconds if negative else seconds


def extract_span(data: dict, unit: str = "s") -> tuple[float, float]:
    """Pull a (start, end) pair out of a segment/word dict.

    Tries, in order: the current server's `start`/`end` floats, the `-oj`-style
    `offsets` block (always milliseconds), and the `timestamps` block (always
    clock strings). Raises KeyError if none are present.
    """
    if "start" in data and "end" in data:
        return parse_timestamp(data["start"], unit), parse_timestamp(data["end"], unit)

    offsets = data.get("offsets")
    if isinstance(offsets, dict) and "from" in offsets and "to" in offsets:
        return parse_timestamp(offsets["from"], "ms"), parse_timestamp(offsets["to"], "ms")

    stamps = data.get("timestamps")
    if isinstance(stamps, dict) and "from" in stamps and "to" in stamps:
        return parse_timestamp(stamps["from"]), parse_timestamp(stamps["to"])

    raise KeyError("no start/end, offsets or timestamps in entry")
