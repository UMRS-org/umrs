from umrs.shared.sql_helpers import create_db_name, metadata_dict


def test_create_db_name_normalizes_spaces_and_case():
    assert create_db_name("SqlUser", "My XP") == "my_xp_sqluser"


def test_create_db_name_appends_username_for_per_user_isolation():
    assert create_db_name("SqlUser", "My XP", "Alice") == "my_xp_sqluser_alice"


def test_create_db_name_distinct_users_distinct_databases():
    a = create_db_name("demo_opossum", "Opossum Analysis", "mnagaso")
    b = create_db_name("demo_opossum", "Opossum Analysis", "mnagaso3")
    assert a != b
    assert a == "opossum_analysis_demo_opossum_mnagaso"
    assert b == "opossum_analysis_demo_opossum_mnagaso3"


def test_create_db_name_without_username_is_legacy_name():
    # Omitting username must reproduce the legacy (un-namespaced) name so
    # migration code can locate pre-existing databases.
    legacy = create_db_name("demo_opossum", "Opossum Analysis")
    assert legacy == "opossum_analysis_demo_opossum"


def test_metadata_dict_preserves_metadata_order():
    sql_cfg = {"meta_data": ["filename", "checksum", "owner"]}

    assert metadata_dict(sql_cfg) == {
        "key0": "filename",
        "key1": "checksum",
        "key2": "owner",
    }
