from dataclasses import dataclass, field
from threading import RLock
<<<<<<< HEAD
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING
from umrs.s3 import count_objects_in_bucket
from typing import Union

if TYPE_CHECKING:
    from umrs.sql import DataBase

=======
from typing import Any, Dict, Optional, Tuple, Union
from umrs.s3 import count_objects_in_bucket
from typing import Union

>>>>>>> 3a30661 (change dictionary store froi client and server to a class for better lisibility)
@dataclass
class ServerBucketEntry:
    """
    Represents a single server bucket entry, storing all metadata, connections, and configuration
    for a given user/bucket combination.
<<<<<<< HEAD
    Must be keyed but path_to_bucket
=======
    Must nut keyed but path_to_bucket
>>>>>>> 3a30661 (change dictionary store froi client and server to a class for better lisibility)
    """
    bucket_alias: str
    s3_username: str
    sql_username: str
<<<<<<< HEAD
    data_base: "DataBase"
=======
    data_base: Any  # Could be a database connection object
>>>>>>> 3a30661 (change dictionary store froi client and server to a class for better lisibility)
    db_name: str
    table_name: str
    model: Any  # Reference to a model object
    store_embeddings_flag: bool
    modality_name: str
    s3_connection: Any  # S3 client or connection object
    metadata: Dict
    xp_name: str
    num_objects: int
    output_path_to_bucket: Optional[str] = None
    reentrant_lock: RLock = field(default_factory=RLock)  # Thread-safe lock for this entry

@dataclass
class ClientBucketEntry:
    """
    Represents a single client bucket entry, storing all client relevant information
    for a given user/bucket_alias combination.
    Must be keyed but bucket_alias
    """
    path_to_bucket: str
    username: str
    xp_name: str
    modality_name: str
    s3_configuration_file: str
    metadata: dict
    sql_username: str
    db_name: str
    output_path_to_bucket: Optional[str] = None

BucketEntry = Union[ServerBucketEntry, ClientBucketEntry]

class BucketsCfg:
    """
    A container for BucketEntry objects keyed by `path_to_bucket`.
    Supports storing entries, retrieving them, filtering fields, and searching by output path.
    """
    def __init__(self):
        self._store: Dict[str, BucketEntry] = {}

    def store(self, key: str, entry: BucketEntry):
        """
        Store a new entry or overwrite an existing one for the given path.
        """
        self._store[key] = entry

    def get(self, key: str, filter: Optional[list[str]] = None) -> Union[BucketEntry, Tuple]:
        """
        Retrieve a full entry or a filtered tuple of values.

        Args:
            path_to_bucket: The key for the bucket entry.
            filter: Optional list of field names. If provided, returns a tuple
                    with the values in the order of this list.

        Returns:
            BucketEntry if no filter is given.
            Tuple of filtered values if filter is provided.
            None if the path does not exist.
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        if filter is None:
            return entry

        # Return a tuple of values according to the filter order
        return tuple(getattr(entry, key) for key in filter if hasattr(entry, key))

    def find_by_output_path(self, output_path: str):
        """
        Search for a bucket entry by its `output_path_to_bucket`.

        Args:
            output_path: The output path string to search for.

        Returns:
            Tuple (path_to_bucket, BucketEntry) if found.
            Tuple (None, None) if not found.
        """
        for path, entry in self._store.items():
            if entry.output_path_to_bucket == output_path:
                return path, entry
        return None, None
    
    def update_num_objects(self, path_to_bucket: str):
        """Update the num_objects field for a given bucket entry."""
        entry = self.get(path_to_bucket)
        if entry is None:
            raise KeyError(f"No entry found for bucket path: {path_to_bucket}")

<<<<<<< HEAD
        data_base = entry.data_base
        table_name = entry.table_name
        # num_objects = count_objects_in_bucket(s3_connection, path_to_bucket)  
        num_objects = data_base.count_s3_objects(table_name)
=======
        s3_connection = entry.s3_connection
        # Replace with your actual function to count objects
        num_objects = count_objects_in_bucket(s3_connection, path_to_bucket)  
>>>>>>> 3a30661 (change dictionary store froi client and server to a class for better lisibility)
        entry.num_objects = num_objects
    
    def build_dict_for_user(self, s3_username: str) -> dict[str, dict[str, Any]]:
        """
        Builds a dictionary for all entries matching the given s3_username.

        Returns a dict where:
        - keys are path_to_bucket
        - values are dicts mapping each field name (as string) to its value
        """
        result = {}
        for path, entry in self._store.items():
            if entry.s3_username == s3_username:
                # Convert dataclass to dictionary
                result[path] = {field.name: getattr(entry, field.name) 
                                for field in entry.__dataclass_fields__.values()}
        return result

<<<<<<< HEAD
    def replace_cfg(self, other: "BucketsCfg"):
        """Replace current store with the contents of another BucketStore."""
        if not isinstance(other, BucketsCfg):
            raise TypeError("other must be a BucketCfg")
        
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot replace {type(self).__name__} "
                f"with {type(other).__name__}"
            )
=======
    def replace_cfg(self, other: "BucketStore"):
        """Replace current store with the contents of another BucketStore."""
        if not isinstance(other, BucketStore):
            raise TypeError("other must be a BucketStore")
>>>>>>> 3a30661 (change dictionary store froi client and server to a class for better lisibility)
        self._store = other._store.copy()

    def items(self):
        """Allow iteration like a dict: for path, entry in bucket_cfg.items()"""
        return self._store.items()
    
    def values(self):
        """Allow iteration over entries only."""
        return self._store.values()
    
    def keys(self):
        """Allow iteration over keys only."""
        return self._store.keys()
    
    def clear(self):
        """Remove all entries from the store."""
        self._store.clear()
<<<<<<< HEAD
 
=======
    
# def store_db_model(db_model_store, path_to_bucket, output_path_to_bucket, bucket_alias, 
#                    s3_username, data_base, db_name, table_name, sql_username, model,
#                    store_embeddings_flag, modality_name, s3_connection, metadata, xp_name,
#                    num_objects, reentrant_lock):
#     """Stores data base and model for a given (user, bucket_alias)."""
#     # if user not in db_model_store:
#     #     storage[user] = {}  # Initialize user entry
    
#     db_model_store[path_to_bucket] = {
#                                         "bucket_alias": bucket_alias,
#                                         "s3_username": s3_username,
#                                         "sql_username": sql_username,
#                                         "data_base": data_base,
#                                         "db_name": db_name,
#                                         "table_name": table_name,
#                                         "model": model,
#                                         "store_embeddings_flag": store_embeddings_flag,
#                                         "modality_name": modality_name,
#                                         "s3_connection": s3_connection,
#                                         "metadata": metadata,
#                                         "xp_name": xp_name,
#                                         "output_path_to_bucket": output_path_to_bucket,
#                                         "num_objects": num_objects,
#                                         "reentrant_lock": reentrant_lock,
#                                         }


# def query_db_model(db_model_store, path_to_bucket):
#     """Retrieves database and model for a given (user, bucket_alias)."""
#     entry = db_model_store.get(path_to_bucket, None)
#     if not entry:
#         return None, None, None, None, None, None, None, None, None, None
#     else:
#         return entry["s3_username"], entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"], entry['db_name'], entry["output_path_to_bucket"], entry["reentrant_lock"]

# def find_entry_by_output_bucket(db_model_store, output_path_to_bucket):
#     for path_to_bucket, entry in db_model_store.items():
#         if entry.get("output_path_to_bucket") == output_path_to_bucket:
#             return path_to_bucket, entry
#     return None, None
>>>>>>> 3a30661 (change dictionary store froi client and server to a class for better lisibility)
