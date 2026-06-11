from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from umrs.server.sql import DataBase


@dataclass
class ServerBucketEntry:
    """Server-side bucket metadata keyed by path_to_bucket."""

    bucket_alias: str
    s3_username: str
    sql_username: str
    data_base: "DataBase"
    db_name: str
    table_name: str
    model: Any
    store_embeddings_flag: bool
    modality_name: str
    task: str
    s3_connection: Any
    metadata: dict
    xp_name: str
    num_objects: int
    return_type: str
    output_path_to_bucket: str | None = None
    reentrant_lock: RLock = field(default_factory=RLock)


@dataclass
class ClientBucketEntry:
    """Client-side bucket metadata keyed by bucket_alias."""

    path_to_bucket: str
    username: str
    xp_name: str
    modality_name: str
    task: str
    s3_configuration_file: str
    metadata: dict
    sql_username: str
    db_name: str
    output_path_to_bucket: str | None = None


BucketEntry = Union[ServerBucketEntry, ClientBucketEntry]


class BucketsCfg:
    """Container for bucket entries used by both client and server flows."""

    def __init__(self):
        self._store: dict[str, BucketEntry] = {}

    def store(self, key: str, entry: BucketEntry):
        self._store[key] = entry

    def get(
        self,
        key: str,
        fields: list[str] | None = None,
        **kwargs: Any,
    ) -> BucketEntry | tuple:
        if fields is None and "filter" in kwargs:
            fields = kwargs["filter"]

        entry = self._store.get(key)
        if entry is None:
            return None

        if fields is None:
            return entry

        return tuple(getattr(entry, item) for item in fields if hasattr(entry, item))

    def find_by_output_path(self, output_path: str):
        for path, entry in self._store.items():
            if entry.output_path_to_bucket == output_path:
                return path, entry
        return None, None

    def update_num_objects(self, path_to_bucket: str):
        entry = self.get(path_to_bucket)
        if entry is None:
            raise KeyError(f"No entry found for bucket path: {path_to_bucket}")

        data_base = entry.data_base
        table_name = entry.table_name
        with entry.reentrant_lock:
            entry.num_objects = data_base.count_s3_objects(table_name)

    def build_dict_for_user(self, s3_username: str) -> dict[str, dict[str, Any]]:
        result = {}
        for path, entry in self._store.items():
            if entry.s3_username == s3_username:
                result[path] = {
                    field_info.name: getattr(entry, field_info.name)
                    for field_info in entry.__dataclass_fields__.values()
                }
        return result

    def replace_cfg(self, other: "BucketsCfg"):
        if not isinstance(other, BucketsCfg):
            raise TypeError("other must be a BucketCfg")

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot replace {type(self).__name__} with {type(other).__name__}"
            )
        self._store = other._store.copy()

    def items(self):
        return self._store.items()

    def values(self):
        return self._store.values()

    def keys(self):
        return self._store.keys()

    def clear(self):
        self._store.clear()
