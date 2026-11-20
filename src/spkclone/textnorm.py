"""Small English text normaliser for the TTS front-end (the pretrained Tacotron2 models expect spelled-out text).

Handles: common abbreviations, & % $ @ +, ordinals (21st), decimals (3.14), comma-grouped and plain integers.
Limitations (documented, not handled): years are read as cardinals ("1990" -> "one thousand nine hundred ninety"),
no dates/times/units/roman numerals, no context-dependent abbreviations (St. = saint/street is left as is).
Numbers are written without hyphens ("twenty one") to keep the character set simple for Tacotron2.
Use `has_unhandled()` to drop evaluation sentences the normaliser cannot fully clean.
"""
import re

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand")]
ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                     "nine": "ninth", "twelve": "twelfth"}
ABBREVIATIONS = [(r"\bMr\.", "mister"), (r"\bMrs\.", "missus"), (r"\bMs\.", "miz"), (r"\bDr\.", "doctor"),
                 (r"\bProf\.", "professor"), (r"\bvs\.", "versus"), (r"\betc\.", "et cetera"),
                 (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\bNo\.(?=\s*\d)", "number")]
SYMBOLS = [("&", " and "), ("@", " at "), ("+", " plus ")]


def _under_1000(n):
    words = []
    if n >= 100:
        words += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(TENS[n // 10])
        if n % 10:
            words.append(ONES[n % 10])
    elif n or not words:
        words.append(ONES[n])
    return words


def num_to_words(n):
    """Non-negative integer < 10**12 -> English words, e.g. 1250 -> 'one thousand two hundred fifty'."""
    n = int(n)
    if n < 0 or n >= 10**12:
        raise ValueError(f"out of range: {n}")
    if n == 0:
        return "zero"
    words = []
    for value, name in SCALES:
        if n >= value:
            words += _under_1000(n // value) + [name]
            n %= value
    if n:
        words += _under_1000(n)
    return " ".join(words)


def ordinal_words(n):
    words = num_to_words(n).split()
    last = words[-1]
    if last in ORDINAL_IRREGULAR:
        words[-1] = ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _decimal_words(int_part, frac_part):
    return f"{num_to_words(int_part)} point " + " ".join(ONES[int(d)] for d in frac_part)


def _amount_words(s):
    s = s.replace(",", "")
    if "." in s:
        a, b = s.split(".", 1)
        return _decimal_words(a or "0", b)
    return num_to_words(s)


def _money(m):
    s = m.group(1).replace(",", "")
    dollars, _, cents = s.partition(".")
    d = int(dollars or 0)
    out = f"{num_to_words(d)} {'dollar' if d == 1 else 'dollars'}"
    if cents and len(cents) == 2 and int(cents):
        c = int(cents)
        out += f" {num_to_words(c)} {'cent' if c == 1 else 'cents'}"
    elif cents and len(cents) != 2:
        out = f"{_decimal_words(dollars or '0', cents)} dollars"
    return out


def normalize(text):
    for pat, rep in ABBREVIATIONS:
        text = re.sub(pat, rep, text)
    text = re.sub(r"\$(\d[\d,]*(?:\.\d+)?)", _money, text)
    text = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", lambda m: f"{_amount_words(m.group(1))} percent", text)
    for sym, rep in SYMBOLS:
        text = text.replace(sym, rep)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal_words(m.group(1)), text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m.group(0).replace(",", ""), text)
    text = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: _decimal_words(m.group(1), m.group(2)), text)
    text = re.sub(r"\b\d+\b", lambda m: num_to_words(m.group(0)), text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


def has_unhandled(text):
    """True if the (normalised) text still contains digits or characters outside letters + basic punctuation."""
    return bool(re.search(r"[^A-Za-z\s,.!?;:'\-\"]", text))
