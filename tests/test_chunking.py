import pytest

from rag_doc_qa.chunking import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_SIZE_TOKENS, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_text_shorter_than_chunk_size_is_a_single_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == ["one two three four five"]


def test_sliding_window_overlap_and_final_partial_window():
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)
    chunk_size, overlap = 10, 3

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) == 4
    # Consecutive windows must share exactly `overlap` tokens at the boundary.
    assert chunks[0].split()[-overlap:] == chunks[1].split()[:overlap]
    assert chunks[1].split()[-overlap:] == chunks[2].split()[:overlap]
    # The final window is a short, non-overlapping-sized remainder.
    assert chunks[-1].split() == words[-4:]


def test_no_overlap_produces_disjoint_chunks():
    words = [f"word{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=5, overlap=0)
    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word5 word6 word7 word8 word9",
        "word10 word11 word12 word13 word14",
        "word15 word16 word17 word18 word19",
    ]


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=5, overlap=-1)


def test_overlap_not_smaller_than_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=5, overlap=5)


def test_default_settings_split_sample_document_into_multiple_chunks():
    from pathlib import Path

    sample_doc_path = Path(__file__).resolve().parent.parent / "data" / "sample_document.txt"
    sample_text = sample_doc_path.read_text(encoding="utf-8")
    chunks = chunk_text(sample_text, DEFAULT_CHUNK_SIZE_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS)
    assert len(chunks) >= 3
    # Rejoining chunks (minus the duplicated overlap) must not lose any source words.
    total_words_with_overlap = sum(len(c.split()) for c in chunks)
    assert total_words_with_overlap >= len(sample_text.split())
