import pytest

from spkclone.textnorm import has_unhandled, normalize, num_to_words, ordinal_words

CASES = [  # (input, expected) - also printed as before/after in the test log
    ("Dr. Smith paid $5 for 3 apples.", "doctor Smith paid five dollars for three apples."),
    ("It costs $2.50.", "It costs two dollars fifty cents."),
    ("It costs $1.", "It costs one dollar."),
    ("Rates rose 12%.", "Rates rose twelve percent."),
    ("Rates rose 12.5%.", "Rates rose twelve point five percent."),
    ("the 21st century", "the twenty first century"),
    ("1,250 people came.", "one thousand two hundred fifty people came."),
    ("pi is 3.14", "pi is three point one four"),
    ("Tom & Jerry", "Tom and Jerry"),
    ("e.g. this one", "for example this one"),
    ("Room No. 5 is free.", "Room number five is free."),
    ("Mr. and Mrs. Brown", "mister and missus Brown"),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize(raw, expected):
    out = normalize(raw)
    print(f"\n  before: {raw}\n  after:  {out}")
    assert out == expected
    assert not has_unhandled(out)


@pytest.mark.parametrize("n,words", [(0, "zero"), (7, "seven"), (13, "thirteen"), (40, "forty"), (99, "ninety nine"),
                                     (105, "one hundred five"), (999, "nine hundred ninety nine"),
                                     (1000, "one thousand"), (1000000, "one million"),
                                     (2000017, "two million seventeen")])
def test_num_to_words(n, words):
    assert num_to_words(n) == words


@pytest.mark.parametrize("n,words", [(1, "first"), (2, "second"), (3, "third"), (12, "twelfth"), (20, "twentieth"),
                                     (21, "twenty first"), (100, "one hundredth")])
def test_ordinals(n, words):
    assert ordinal_words(n) == words


def test_has_unhandled():
    assert not has_unhandled("Hello, world. It's fine!")
    assert has_unhandled("Call 911")
    assert has_unhandled("costs 5 euros")
    assert has_unhandled("50 °C")


def test_normalising_before_splitting_keeps_abbreviations_together():
    from spkclone.tts import split_text

    assert split_text(normalize("Dr. Smith arrived. He sat."), max_chars=10) == ["doctor Smith arrived.", "He sat."]
