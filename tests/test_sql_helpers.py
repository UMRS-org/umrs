from umrs.shared.sql_helpers import create_db_name, metadata_dict


def test_create_db_name_normalizes_spaces_and_case():
    assert create_db_name("SqlUser", "My XP") == "my_xp_sqluser"


def test_metadata_dict_preserves_metadata_order():
    sql_cfg = {"meta_data": ["filename", "checksum", "owner"]}

    assert metadata_dict(sql_cfg) == {
        "key0": "filename",
        "key1": "checksum",
        "key2": "owner",
    }