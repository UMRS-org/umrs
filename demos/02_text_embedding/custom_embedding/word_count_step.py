"""Example custom embedding step — word-count vector.

Demonstrates how to create a simple text embedding using the Pipeline SDK.
This step converts text files into word-count vectors, showing the
user_script:// pathway for custom embeddings.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np

from umrs.pipeline_sdk import Step, StepContext, StepResult


class WordCountEmbeddingStep(Step):
    """Convert text files into fixed-size word-count vectors."""

    name = "word_count_embedding"

    def setup(self, config: dict) -> None:
        self.vector_size = config.get("vector_size", 128)
        self.output_prefix = config.get("output_prefix", "embeddings/")

    def run(self, ctx: StepContext) -> StepResult:
        all_vectors = []
        output_keys = []
        checksums = []
        metadata_list = []

        for key in ctx.input_keys:
            # Download text
            raw = ctx.download_bytes(key)
            text = raw.decode("utf-8", errors="replace")

            # Tokenize and count
            words = re.findall(r"\w+", text.lower())
            counts = Counter(words)

            # Build fixed-size vector by hashing word → bucket
            vector = np.zeros(self.vector_size, dtype=np.float32)
            for word, count in counts.items():
                bucket = hash(word) % self.vector_size
                vector[bucket] += count

            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            all_vectors.append(vector)

            # Upload embedding as .npy
            emb_key = f"{self.output_prefix}{key}.npy"
            ctx.upload_numpy(emb_key, vector)
            output_keys.append(emb_key)

            # Upload metadata
            head = ctx.head_object(key)
            checksums.append(head["ETag"].strip('"'))
            metadata_list.append(
                {
                    "source_key": key,
                    "word_count": len(words),
                    "unique_words": len(counts),
                }
            )

        return StepResult(
            output_keys=output_keys,
            embeddings=np.stack(all_vectors) if all_vectors else None,
            checksums=checksums,
            metadata=metadata_list,
        )
