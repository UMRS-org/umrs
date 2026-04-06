"""Custom step: count words and characters in text files."""

class WordCounter:
    name = "count_words"

    def setup(self, config):
        self.min_word_length = config.get("min_word_length", 1)

    def teardown(self):
        pass

    def run(self, ctx):
        results = {}
        for key in ctx.input_keys:
            if key.endswith(".txt"):
                data = ctx.download_bytes(key)
                text = data.decode("utf-8", errors="replace")
                words = [w for w in text.split() if len(w) >= self.min_word_length]
                results[key] = {
                    "total_words": len(text.split()),
                    "filtered_words": len(words),
                    "characters": len(text),
                    "lines": text.count("\n") + 1,
                }

        import json
        summary = json.dumps(results, indent=2)
        ctx.upload_bytes("word_count_results.json", summary.encode())

        return type("R", (), {
            "output_keys": ["word_count_results.json"],
            "embeddings": None,
            "checksums": {},
            "metadata": {"summary": results},
        })()
