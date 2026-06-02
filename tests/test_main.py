import pytest
from main import _merge_dicts, can_fetch

def test_merge_dicts():
    assert _merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert _merge_dicts(None, {"b": 2}) == {"b": 2}
    assert _merge_dicts({"a": 1}, None) == {"a": 1}
    assert _merge_dicts([], {"b": 2}) == {"b": 2}

def test_can_fetch():
    assert can_fetch("https://google.com/", "test-agent") == True

