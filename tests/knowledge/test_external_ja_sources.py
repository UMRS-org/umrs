"""D5 — JA-native external source connectors (CiNii / J-STAGE / WEKO3)."""

from __future__ import annotations

from unittest.mock import patch

from umrs.knowledge.external import (
    CiNiiSource,
    ExternalKnowledgeSearcher,
    ExternalSourceType,
    JStageSource,
    SearchStatus,
    WEKO3Source,
    search_cinii,
    search_jstage,
    search_weko3,
)

CINII_JSON = """{
  "@graph": [{
    "items": [
      {
        "title": "量子ドットにおけるスピン緩和機構",
        "creator": [{"name": "田中 信樹"}],
        "description": "量子ドットの緩和機構を議論する。",
        "pubDate": "2024-03-01",
        "link": {"@id": "https://cir.nii.ac.jp/crid/1050001"}
      },
      {
        "title": "Superconductivity in FeSe thin films",
        "creator": [{"name": "A. Yamamoto"}],
        "description": "",
        "pubDate": "2015-11",
        "link": {"@id": "https://cir.nii.ac.jp/crid/1050002"}
      }
    ]
  }]
}"""

JSTAGE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>機械学習を用いた地動ポテンシャル予測</title>
    <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">佐藤 花子</dc:creator>
    <dc:description xmlns:dc="http://purl.org/dc/elements/1.1/">地動ポテンシャル予測を報告する。</dc:description>
    <prism:publicationDate xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/">2025-06</prism:publicationDate>
    <link href="https://www.jstage.jst.go.jp/article/jsrt/12/3/123/_article" />
  </entry>
</feed>
"""

OAI_DC_XML = """<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>球面調和関数に基づく全地球大気モデル</dc:title>
          <dc:creator>鈴木 一郎</dc:creator>
          <dc:description>We describe an atmospheric global model.</dc:description>
          <dc:date>2023-08</dc:date>
          <dc:identifier>https://repo.nii.ac.jp/record/1234</dc:identifier>
        </oai_dc:dc>
      </metadata>
    </record>
    <record>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                   xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Unrelated paper</dc:title>
          <dc:creator>X. Y.</dc:creator>
        </oai_dc:dc>
      </metadata>
    </record>
  </ListRecords>
</OAI-PMH>
"""


def _patch_fetch(expected: str):
    return (
        patch.object(CiNiiSource, "_fetch_url", return_value=expected),
        patch.object(JStageSource, "_fetch_url", return_value=expected),
        patch.object(WEKO3Source, "_fetch_url", return_value=expected),
    )


class TestCiNiiSource:
    def test_init(self):
        s = CiNiiSource()
        assert s.BASE_URL.startswith("https://cir.nii.ac.jp")

    def test_search_parses_json_items(self):
        s = CiNiiSource()
        with patch.object(s, "_fetch_url", return_value=CINII_JSON):
            r = s.search("superconductivity FeSe")
        assert r.status == SearchStatus.SUCCESS
        assert r.source_type == ExternalSourceType.CINII
        assert r.total_results == 2
        fe_se = [ref for ref in r.references if "FeSe" in ref.title][0]
        assert fe_se.authors == ["A. Yamamoto"]
        assert fe_se.url == "https://cir.nii.ac.jp/crid/1050002"

    def test_ja_title_preserved(self):
        s = CiNiiSource()
        with patch.object(s, "_fetch_url", return_value=CINII_JSON):
            r = s.search("spin relaxation")
        ja = [ref for ref in r.references if "量子" in ref.title][0]
        assert ja.title == "量子ドットにおけるスピン緩和機構"

    def test_error_on_fetch_failure(self):
        s = CiNiiSource()
        with patch.object(s, "_fetch_url", return_value=None):
            r = s.search("x")
        assert r.status == SearchStatus.ERROR

    def test_convenience_fn(self):
        s = CiNiiSource()
        with patch.object(CiNiiSource, "_fetch_url", return_value=CINII_JSON):
            r = search_cinii("x")
        assert r.status == SearchStatus.SUCCESS


class TestJStageSource:
    def test_init(self):
        s = JStageSource()
        assert "jstage.jst.go.jp" in s.BASE_URL

    def test_search_parses_atom_entries(self):
        s = JStageSource()
        with patch.object(s, "_fetch_url", return_value=JSTAGE_XML):
            r = s.search("機械学習")
        assert r.status == SearchStatus.SUCCESS
        assert r.source_type == ExternalSourceType.JSTAGE
        assert r.total_results == 1
        ref = r.references[0]
        assert ref.title == "機械学習を用いた地動ポテンシャル予測"
        assert ref.authors == ["佐藤 花子"]
        assert "jsrt" in ref.url

    def test_error_on_fetch_failure(self):
        s = JStageSource()
        with patch.object(s, "_fetch_url", return_value=None):
            r = s.search("x")
        assert r.status == SearchStatus.ERROR

    def test_convenience_fn(self):
        with patch.object(JStageSource, "_fetch_url", return_value=JSTAGE_XML):
            r = search_jstage("機械学習")
        assert r.status == SearchStatus.SUCCESS


class TestWEKO3Source:
    def test_init(self):
        s = WEKO3Source()
        assert hasattr(s, "search")

    def test_search_filters_by_title_match(self):
        s = WEKO3Source()
        with patch.object(s, "_fetch_url", return_value=OAI_DC_XML):
            r = s.search("球面調和")
        assert r.status == SearchStatus.SUCCESS
        assert r.source_type == ExternalSourceType.WEKO3
        assert r.total_results == 1
        ref = r.references[0]
        assert ref.title == "球面調和関数に基づく全地球大気モデル"
        assert ref.authors == ["鈴木 一郎"]
        assert ref.url.startswith("https://")

    def test_no_match_no_results(self):
        s = WEKO3Source()
        with patch.object(s, "_fetch_url", return_value=OAI_DC_XML):
            r = s.search("quantum dots")
        assert r.status == SearchStatus.NO_RESULTS

    def test_max_results_honored(self):
        s = WEKO3Source()
        with patch.object(s, "_fetch_url", return_value=OAI_DC_XML):
            r = s.search("", max_results=1)
        assert len(r.references) <= 1

    def test_custom_base_url(self):
        s = WEKO3Source()
        with patch.object(s, "_fetch_url", return_value=OAI_DC_XML) as mock:
            r = s.search("anything", base_url="https://my-uni.ac.jp/oai")
        args = mock.call_args.args
        assert "my-uni.ac.jp" in args[0]

    def test_error_on_fetch_failure(self):
        s = WEKO3Source()
        with patch.object(s, "_fetch_url", return_value=None):
            r = s.search("x")
        assert r.status == SearchStatus.ERROR

    def test_convenience_fn_default_base_url(self):
        with patch.object(WEKO3Source, "_fetch_url", return_value=OAI_DC_XML):
            r = search_weko3("球面調和")
        assert r.status == SearchStatus.SUCCESS


class TestSearcherIntegration:
    def test_default_searcher_includes_ja_sources(self):
        searcher = ExternalKnowledgeSearcher()
        assert ExternalSourceType.CINII in searcher._sources
        assert ExternalSourceType.JSTAGE in searcher._sources
        # WEKO3 requires base_url at call time — excluded from search_all default
        assert ExternalSourceType.WEKO3 not in searcher._sources

    def test_single_source_search_ja(self):
        searcher = ExternalKnowledgeSearcher()
        with patch.object(CiNiiSource, "_fetch_url", return_value=CINII_JSON):
            r = searcher.search("x", ExternalSourceType.CINII)
        assert r.status == SearchStatus.SUCCESS
