"""
Text normalization for the Business Entity Resolution challenge.

Every rule here is backed by a specific example seen in the real train data
(see comments). Nothing here uses external data or lookups.
"""
import re
import unicodedata

# ---------------------------------------------------------------- prefixes / junk
# ">> AROHA CHARITABLE..." / "M/s Akar India..." / leading quote/punct junk
_RE_LEADING_JUNK = re.compile(r"^[\s>*\-_.'\"]+")
_RE_MS_PREFIX = re.compile(r"^(?:m/s\.?|messrs\.?)\s+", re.I)

# ---------------------------------------------------------------- alias markers
# "Rizazeph f/k/a Future Foundation Pvt Ltd"  -> two real names for one entity
_RE_ALIAS = re.compile(
    r"\b(?:f/?k/?a|a/?k/?a|d/?b/?a|dba|t/?a|formerly known as|formerly|trading as)\b",
    re.I,
)

# ---------------------------------------------------------------- domain / phone
# "premierconsultancy.com", "hefinance.com", "vatecify.com - 2199670374", "VMCVERDE.COM"
_RE_DOMAIN = re.compile(r"^\s*([\w-]+)\.(com|net|org|in|co|io|biz|info|fr)\b", re.I)
_RE_TRAILING_PHONE = re.compile(r"\s*-\s*\d{7,}\s*$")

# ---------------------------------------------------------------- bracket / id noise
# "[INCORPORATED] PEAK TRADIN6...", "Memorial Association Co (ID: 16503)"
_RE_BRACKETS = re.compile(r"[\[\]{}]")
_RE_PAREN_ID = re.compile(r"\(\s*id\s*[:#]?\s*[\w-]+\s*\)", re.I)
_RE_PAREN = re.compile(r"\([^)]*\)")

# ---------------------------------------------------------------- misc noise tokens
# "504 I-WING, N/A, MUMBAI..."  /  "##847 OXBOW RD"  /  "NULL"
_RE_NULL_TOKEN = re.compile(r"\b(?:null|n/?a|nan)\b", re.I)
_RE_HASH = re.compile(r"#+")

# legal-suffix words that are high-frequency and low-signal on their own
# (kept only for the SQUASHED/no-suffix comparison form; raw text is preserved separately)
_LEGAL_WORDS = {
    "private", "pvt", "ltd", "limited", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llp", "plc", "gmbh",
    # French — no training pairs exist to mine these, so they are hand-listed
    "sarl", "sas", "sasu", "eurl", "snc", "sci",
}

# a duplicated adjacent token: "Akar Akar India", "National National"
_RE_DUP_TOKEN = re.compile(r"\b(\w+)\b(\s+\1\b)+", re.I)

_ADDR_JUNK_RE = re.compile(
    r"\b(?:null|n/?a|nan)\b|#+", re.I
)


def fold_accents(text: str) -> str:
    """'Índia' -> 'India', 'Vanguard Pátriot' -> 'Vanguard Patriot' (Latin accents only)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def collapse_duplicate_tokens(text: str) -> str:
    return _RE_DUP_TOKEN.sub(r"\1", text)


def strip_leading_junk(text: str) -> str:
    text = _RE_LEADING_JUNK.sub("", text)
    text = _RE_MS_PREFIX.sub("", text)
    return text


def extract_paren_id(text: str):
    """Pull out '(ID: 16503)'-style markers; return (text_without_marker, id_or_None)."""
    m = _RE_PAREN_ID.search(text)
    if m:
        return (text[: m.start()] + text[m.end():]).strip(), m.group(0)
    return text, None


def strip_domain_and_phone(name: str):
    """
    'vatecify.com - 2199670374' -> core='vatecify', is_domain=True, has_phone=True
    'premierconsultancy.com'    -> core='premierconsultancy', is_domain=True
    Anything else               -> core=name unchanged, is_domain=False
    """
    has_phone = bool(_RE_TRAILING_PHONE.search(name))
    name = _RE_TRAILING_PHONE.sub("", name)
    m = _RE_DOMAIN.match(name.strip())
    if m:
        return m.group(1), True, has_phone
    return name, False, has_phone


def split_alias(name: str):
    """
    'Rizazeph f/k/a Future Foundation Pvt Ltd' -> ['Rizazeph', 'Future Foundation Pvt Ltd']
    Compare a candidate against BOTH halves when an alias marker is present —
    a plain single-name comparison will silently score this as unrelated.
    """
    parts = _RE_ALIAS.split(name)
    parts = [p.strip(" ,;-") for p in parts if p.strip(" ,;-")]
    return parts if len(parts) > 1 else [name]


def strip_bracket_noise(text: str) -> str:
    text, _id = extract_paren_id(text)
    text = _RE_PAREN.sub(" ", text)
    text = _RE_BRACKETS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_name(raw: str) -> dict:
    """Return every representation a matcher needs, plus flags worth keeping as features."""
    text = raw.strip()
    text = strip_leading_junk(text)
    core, is_domain, has_phone = strip_domain_and_phone(text)
    aliases = split_alias(core if is_domain else text)
    out = []
    for a in aliases:
        a = strip_bracket_noise(a)
        a = fold_accents(a)
        a = collapse_duplicate_tokens(a)
        casefolded = re.sub(r"\s+", " ", a).strip().casefold()
        squashed = re.sub(r"[^a-z0-9]", "", casefolded)
        no_suffix = " ".join(t for t in casefolded.split() if t.strip(".,") not in _LEGAL_WORDS)
        out.append({
            "raw": a,
            "casefolded": casefolded,
            "squashed": squashed,          # catches 'x.com'-style / spacing noise
            "no_suffix": no_suffix.strip() or casefolded,
        })
    return {
        "variants": out,          # >1 only when an f/k/a-style alias was found
        "is_domain_like": is_domain,
        "had_phone_suffix": has_phone,
        "had_alias_marker": len(aliases) > 1,
    }


def clean_address(raw: str) -> dict:
    text = raw.strip()
    text = strip_leading_junk(text)
    text, id_marker = extract_paren_id(text)
    had_null = bool(_RE_NULL_TOKEN.search(text))
    had_hash = "#" in text
    text = _RE_NULL_TOKEN.sub(" ", text)
    text = _RE_HASH.sub("", text)
    text = strip_bracket_noise(text)
    text = fold_accents(text)
    casefolded = re.sub(r"\s+", " ", text).strip().casefold()
    numeric_tokens = re.findall(r"\d+", casefolded)
    word_tokens = frozenset(re.findall(r"[a-z]{2,}", casefolded))  # order-independent bag
    return {
        "raw": raw.strip(),
        "casefolded": casefolded,
        "numeric_tokens": numeric_tokens,
        "word_tokens": word_tokens,
        "is_empty": casefolded == "",
        "had_null_token": had_null,
        "had_hash": had_hash,
        "had_id_marker": id_marker is not None,
    }


def numeric_suffix_overlap(nums_a, nums_b) -> float:
    """
    Handles the leading-digit-drop noise seen repeatedly:
      '504' vs '04'   (dropped '5')
      '2922' vs '922' (dropped '2')
      '1/576' vs '2/576' vs '576' (first digit corrupted/dropped entirely)
    Two numeric tokens 'match' if one is a suffix of the other (length >= 2),
    which is exactly the corruption pattern observed — a strict equality check
    would treat all of the examples above as non-matches.
    Returns the fraction of the SHORTER token list that finds such a match.
    """
    if not nums_a or not nums_b:
        return 0.0
    hits = 0
    shorter, longer = (nums_a, nums_b) if len(nums_a) <= len(nums_b) else (nums_b, nums_a)
    for x in shorter:
        if len(x) < 2:
            continue
        if any(len(y) >= 2 and (x.endswith(y) or y.endswith(x)) for y in longer):
            hits += 1
    return hits / max(len(shorter), 1)


if __name__ == "__main__":
    # quick self-check against real examples from groups.txt
    cases = [
        "Rizazeph f/k/a Future Foundation Pvt Ltd",
        ">> AROHA CHARITABLE CORPORATION CENTER",
        "vatecify.com - 2199670374",
        "M/s Akar India Pvt Ltd.",
        "Akar Akar India Pvt",
        "Memorial Association  Co (ID: 16503)",
        "Vanguard Pátriot National",
    ]
    for c in cases:
        print(c, "->", clean_name(c))

    a = clean_address("504 I-WING, N/A, MUMBAI, THANA ,MAHARASHTRA, Maharashtra")
    b = clean_address("Next Ton Cooton Exch Bldg, 04 I-wing, MH, Mumbai, Nehru Nagar, Kurla (East)Kurla Ii")
    print("\nnumeric suffix overlap:", numeric_suffix_overlap(a["numeric_tokens"], b["numeric_tokens"]))
    print("word overlap:", len(a["word_tokens"] & b["word_tokens"]), a["word_tokens"] & b["word_tokens"])
