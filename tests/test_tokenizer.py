import pytest

from llm.tokenizer import EOT, BPETokenizer, _SPLIT

CORPUS = [
    "the quick brown fox jumps over the lazy dog. " * 50,
    "the theory of the thing is that there are three themes.\n" * 50,
    "naïve café — 日本語のテキスト 🙂 snake_case x = 3.14159;\n\n\ttabbed" * 10,
]

SAMPLES = [
    "",
    "the quick brown fox",
    "  leading and trailing  ",
    "line one\r\nline two\n\n\nline three",
    "naïve café — 日本語 🙂",
    "__init__(self, x_1=42)  # comment!?",
    "unseen: zzzqqq ÆØÅ \x00\x7f",
]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=400)


@pytest.mark.parametrize("text", SAMPLES)
def test_pretokenizer_is_lossless(text):
    assert "".join(_SPLIT.findall(text)) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip(tok, text):
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text


def test_learns_merges_and_compresses(tok):
    assert 256 < tok.vocab_size <= 400
    text = "the quick brown fox jumps over the lazy dog."
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_training_is_deterministic():
    assert BPETokenizer.train(CORPUS, 350).merges == BPETokenizer.train(CORPUS, 350).merges


def test_min_frequency_stops_early():
    tok = BPETokenizer.train(["abcdefg"], vocab_size=1000)
    assert tok.merges == []
    assert tok.vocab_size == 257


def test_special_tokens(tok):
    text = f"hello{EOT}world"
    plain = tok.encode(text)
    assert tok.eot_id not in plain
    assert tok.decode(plain) == text

    special = tok.encode(text, allow_special=True)
    assert special.count(tok.eot_id) == 1
    assert tok.eot_id == tok.vocab_size - 1
    assert tok.decode(special) == text
    assert tok.decode(special, skip_special=True) == "helloworld"


def test_save_load(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.vocab_size == tok.vocab_size
    for text in SAMPLES:
        assert loaded.encode(text) == tok.encode(text)


def test_decode_rejects_out_of_range(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


def test_decode_tolerates_truncated_utf8(tok):
    ids = tok.encode("é")
    assert isinstance(tok.decode(ids[:1]), str)
