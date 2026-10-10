from umrs.shared.utils import flatten_text_chunks


class _WhitespaceTokenizer:
    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        input_ids = []
        offsets = []
        cursor = 0

        for index, token in enumerate(text.split()):
            start = text.index(token, cursor)
            end = start + len(token)
            cursor = end
            input_ids.append(index)
            offsets.append((start, end))

        payload = {"input_ids": input_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return payload


def test_flatten_text_chunks_falls_back_when_nlp_is_missing():
    tokenizer = _WhitespaceTokenizer()

    flat_chunks, chunks_by_text = flatten_text_chunks(
        ["alpha beta gamma delta epsilon"],
        nlp=None,
        tokenizer=tokenizer,
        max_tokens=2,
    )

    assert flat_chunks == ["alpha beta", "gamma delta", "epsilon"]
    assert [chunk["num_tokens"] for chunk in chunks_by_text[0]] == [2, 2, 1]
    assert [chunk["start"] for chunk in chunks_by_text[0]] == [0, 11, 23]
    assert [chunk["end"] for chunk in chunks_by_text[0]] == [10, 22, 30]


def test_flatten_text_chunks_keeps_empty_text_without_nlp():
    tokenizer = _WhitespaceTokenizer()

    flat_chunks, chunks_by_text = flatten_text_chunks(
        [""],
        nlp=None,
        tokenizer=tokenizer,
        max_tokens=2,
    )

    assert flat_chunks == [""]
    assert chunks_by_text == [
        [
            {
                "text": "",
                "start": 0,
                "end": 0,
                "num_tokens": 0,
                "md5": "d41d8cd98f00b204e9800998ecf8427e",
            }
        ]
    ]


def test_fixed_v0_is_the_unchanged_window_chunker():
    """ADR-093 D-11: ``fixed-v0`` (the default scheme) IS the shared window
    chunker — same chunks, same offsets, no header."""
    from umrs.knowledge.chunking import chunk_text
    from umrs.shared.utils import _chunk_text_without_nlp

    tokenizer = _WhitespaceTokenizer()
    text = "alpha beta. Gamma delta epsilon. Zeta"
    expected = _chunk_text_without_nlp(text, tokenizer=tokenizer, max_tokens=2)
    assert chunk_text(text, tokenizer=tokenizer, max_tokens=2) == expected
    assert chunk_text(text, scheme="fixed-v0", tokenizer=tokenizer, max_tokens=2) == expected
    assert not any("header" in chunk for chunk in expected)
