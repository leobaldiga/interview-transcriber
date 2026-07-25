import pytest

from engines.timestamps import extract_span, parse_timestamp


@pytest.mark.parametrize(
    "value,expected",
    [
        (12.34, 12.34),
        (0, 0.0),
        (-1.5, -1.5),
        ("12.34", 12.34),
        ("00:00:12,340", 12.34),
        ("00:00:12.340", 12.34),
        ("01:02:03,500", 3723.5),
        ("1:02:03.5", 3723.5),
        ("02:07", 127.0),
        ("00:00:00,000", 0.0),
        ("-00:00:01,500", -1.5),
        ("  00:00:05,000  ", 5.0),
        ("00:59:59,999", 3599.999),
    ],
)
def test_parse_seconds(value, expected):
    assert parse_timestamp(value) == pytest.approx(expected)


@pytest.mark.parametrize("value,expected", [(12340, 12.34), (12340.0, 12.34), ("12340", 12.34)])
def test_parse_milliseconds(value, expected):
    assert parse_timestamp(value, "ms") == pytest.approx(expected)


def test_clock_strings_ignore_the_unit_hint():
    """A clock string carries its own unit — "ms" must not divide it again."""
    assert parse_timestamp("00:00:12,340", "ms") == pytest.approx(12.34)


@pytest.mark.parametrize("value", [None, True, "", "   ", "not a time", [], {}, "00:00:xx"])
def test_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_rejects_unknown_unit():
    with pytest.raises(ValueError):
        parse_timestamp(1.0, "minutes")


def test_extract_span_prefers_float_seconds():
    assert extract_span({"start": 1.5, "end": 2.5}) == (1.5, 2.5)


def test_extract_span_reads_oj_offsets_as_milliseconds():
    entry = {"offsets": {"from": 1500, "to": 2500}, "timestamps": {"from": "00:00:01,500", "to": "00:00:02,500"}}
    assert extract_span(entry) == (1.5, 2.5)


def test_extract_span_falls_back_to_clock_strings():
    assert extract_span({"timestamps": {"from": "00:01:00,000", "to": "00:01:30,250"}}) == (60.0, 90.25)


def test_extract_span_raises_when_absent():
    with pytest.raises(KeyError):
        extract_span({"text": "hello"})
