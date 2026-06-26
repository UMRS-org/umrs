"""Tests for data catalog path encoding."""

from umrs.data_catalog.paths import (
    decode_s3_key,
    encode_path_component,
    encode_s3_key,
    logical_path_for_object,
    meta_object_key,
)


def test_encode_s3_key_roundtrip():
    key = "proj/uploads/uuid_file name.pdf"
    encoded = encode_s3_key(key)
    assert "/" not in encoded or "%2F" in encoded
    assert decode_s3_key(encoded) == key


def test_meta_object_key():
    assert meta_object_key("a/b.txt").startswith("_data_catalog/objects/")
    assert meta_object_key("a/b.txt").endswith(".meta.json")


def test_logical_path_for_object():
    path = logical_path_for_object("my project", "my project/uploads/x.csv", "structured")
    assert path.startswith("my%20project/")
    assert path.endswith("structured/")


def test_encode_path_component_special_chars():
    assert encode_path_component("a/b") == "a%2Fb"
