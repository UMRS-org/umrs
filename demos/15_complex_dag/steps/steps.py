"""Step implementations for the complex DAG demo."""


class LoadData:
    name = "load_data"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        data = {"texts": ["hello world", "complex analysis"], "meta": {"source": "demo"}}
        ctx.metadata["loaded_data"] = data
        return type(
            "R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": data}
        )()


class AnalyzeText:
    name = "analyze_text"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        data = ctx.metadata.get("loaded_data", {})
        texts = data.get("texts", [])
        analysis = {"word_count": sum(len(t.split()) for t in texts), "doc_count": len(texts)}
        ctx.metadata["text_analysis"] = analysis
        return type(
            "R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": analysis}
        )()


class AnalyzeMetadata:
    name = "analyze_metadata"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        data = ctx.metadata.get("loaded_data", {})
        meta = data.get("meta", {})
        result = {"field_count": len(meta), "source": meta.get("source", "unknown")}
        ctx.metadata["meta_analysis"] = result
        return type(
            "R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": result}
        )()


class MergeResults:
    name = "merge_results"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        merged = {
            "text": ctx.metadata.get("text_analysis", {}),
            "meta": ctx.metadata.get("meta_analysis", {}),
        }
        ctx.metadata["merged"] = merged
        return type(
            "R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": merged}
        )()


class GenerateReport:
    name = "generate_report"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        merged = ctx.metadata.get("merged", {})
        report = f"=== Analysis Report ===\nText: {merged.get('text', {})}\nMeta: {merged.get('meta', {})}\n"
        ctx.metadata["report"] = report
        return type(
            "R",
            (),
            {
                "output_keys": [],
                "embeddings": None,
                "checksums": {},
                "metadata": {"report": report},
            },
        )()
