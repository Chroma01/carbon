"""SeqLocate verifier / RL reward (v0.4): verify(reply, gold_answer_json, mode="strict") -> 1.0 or 0.0. Never raises.

Final answer
- Either exactly one <answer>...</answer> block (lowercase tag), or, when no tag is used, a final "Answer: X" line (the
  basic-DNA format). Tolerated: dash-separated bases (A-C-G-T), lowercase tag (any other-case <ANSWER> tag, a second
  block, or an unclosed tag scores 0). Replies over 100,000 characters score 0. Parsing is linear-time (no backtracking
  regex over the reply).
- Task-specific exact grammar. Tolerated: surrounding whitespace, a pair of quotes / backticks, ONE trailing period,
  and lowercase DNA / bases / amino acids. Numbers are bounded (<= 9 digits; thousands separators allowed). Nothing
  else is stripped: an extra N, a trailing word, a second number, a missing required field all score 0.
- Several-valid-answer tasks use predicates: diffs by re-applying the stated edits with minimal events only (diff:
  substitution 1 base, or an insertion / deletion of 1-3 bases; multi_snv: single-base substitutions; REF != ALT;
  coordinates in bounds; no duplicate positions), synonymous edits / site removal / designs by their constraints,
  position lists as exact sets.

Reasoning (mode="strict"): every statement written in one of these shapes must be true. These are the shapes the SFT
targets use, so a model trained on them is checked on all of its stated facts; other prose is not verified.

- interval: "bases A-B are SEQ", "positions A to B read SEQ", optional "(1-based)" / "(1 based)" / "(0-based)"
- line: "line starting at S (SEQ)"
- ordinal: "... and position P is the Nth of these" / "the Nth base of that line" (N = P - block start + 1)
- point: "position P is B", "base P is B"
- b-side: "position P is X in A and Y in B"
- codon: "codon K: bases A-B ..." / "codon K is the first stop: bases A-B ..." (K = (A - 1) / 3 + 1); "XYZ, which
  encodes Q" / "XYZ also encodes Q" / "has XYZ (Q)"; "keeps codon K as Q"
- length: "the sequence is N bases long" / "has N bases"; "(N bases including the stop codon)" with the interval
- counts: "bases A-B contain N" (the base(s) the task counts); "N G or C (P%)" for designs (own sequence)
- occurrence: "SEQ occurs [exactly] once | twice | N times"
- arithmetic (revstrand_pos / genomic_pos): every number in the reasoning must be one of the task's numbers

Claims about positions may be true in the original OR in the edited / second sequence of the task (A or B, reference or
clone, before or after an edit). Untagged coordinates use the prompt's convention.

Guessable tasks (base_at, check_base): strict mode also requires a true interval or line claim of >= 2 bases covering
the queried position (restating the answer as "position P is B" does not count).

mode="answer": final answer only (same one-block rule and grammar).
"""

import json
import re


CODON = {
    a + b + c: aa
    for (a, b, c), aa in zip(
        [(x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"],
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
        strict=False,
    )
}
COMP = str.maketrans("ACGT", "TGCA")
AMINO = set("ACDEFGHIKLMNPQRSTVWY")
MAX_REPLY = 100_000
N = r"(\d{1,3}(?:,\d{3}){1,2}|\d{1,9})"  # bounded integer (no 4,300-digit int() crash)
ORD = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def translate(s):
    return "".join(CODON.get(s[i : i + 3], "X") for i in range(0, len(s) - len(s) % 3, 3))


def revcomp(s):
    return s.translate(COMP)[::-1]


def occurrences(seq, motif):
    """1-based starts of all (overlapping) occurrences."""
    out, i = [], seq.find(motif) if motif else -1
    while i >= 0:
        out.append(i + 1)
        i = seq.find(motif, i + 1)
    return out


def to_int(x):
    return int(x.replace(",", ""))


# ---------------- answer extraction (linear time) ----------------
ANSWER_LINE = re.compile(r"(?im)^\s*(?:\*\*)?answer(?:\*\*)?\s*:\s*(.+?)\s*$")


def final_answers(reply):
    """Contents of <answer> blocks, or of 'Answer: ...' lines when no tag is used. None if the tag structure is invalid."""
    low = reply.lower()
    if low.count("<answer>") != reply.count("<answer>") or low.count("</answer>") != reply.count("</answer>"):
        return None  # <ANSWER> etc.
    out, i = [], 0
    while True:
        s = reply.find("<answer>", i)
        if s < 0:
            break
        e = reply.find("</answer>", s + 8)
        if e < 0:
            return None
        inner = reply[s + 8 : e]
        if "<answer>" in inner:
            return None
        out.append(inner)
        i = e + 9
    if reply.count("</answer>") != len(out):
        return None
    if out:
        return out
    lines = ANSWER_LINE.findall(reply)
    return lines if lines else []


def unwrap(ans):
    a = ans.strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "'\"`":
        a = a[1:-1].strip()
    if a.endswith(".") and not a.endswith(".."):
        a = a[:-1].rstrip()
    return a


def as_dna(a):
    s = re.sub(r"[\s-]+", "", a).upper()  # A-C-G-T and line breaks allowed
    return s if re.fullmatch(r"[ACGT]+", s) else None


def as_int(a):
    return to_int(a) if re.fullmatch(N, a) else None


def apply_edit(a, pos, ref, alt):
    """Apply 'pos ref>alt' (1-based on a; '-' = empty). None if malformed or out of bounds."""
    ref, alt = ref.replace("-", ""), alt.replace("-", "")
    if not ref and not alt:
        return None
    if ref:
        if not (1 <= pos <= len(a) - len(ref) + 1) or a[pos - 1 : pos - 1 + len(ref)] != ref:
            return None
        return a[: pos - 1] + alt + a[pos - 1 + len(ref) :]
    if not (0 <= pos <= len(a)):
        return None
    return a[:pos] + alt + a[pos:]


EVENT = re.compile(N + r"\s*([ACGT]{1,3}|-)\s*>\s*([ACGT]{1,3}|-)")


def parse_events(a):
    """'57 T>C; 1,234 G>A' -> [(57,'T','C'), (1234,'G','A')]; events separated by ';' or ', ' (thousands commas allowed)."""
    parts = [p.strip() for p in re.split(r";|,\s+(?=\d)|\n", a.upper()) if p.strip()]
    out = []
    for p in parts:
        m = EVENT.fullmatch(p)
        if not m:
            return None
        out.append((to_int(m.group(1)), m.group(2), m.group(3)))
    return out


def minimal(ev, single_snv):
    pos, ref, alt = ev
    if ref == alt:
        return False
    if single_snv:
        return len(ref) == 1 and len(alt) == 1 and ref != "-" and alt != "-"
    if ref != "-" and alt != "-":
        return len(ref) == 1 and len(alt) == 1  # substitution: one base
    return True  # 1-3 base insertion or deletion (EVENT caps at 3)


# ---------------- typed claims ----------------
CONV = r"(?:\s*\((1[- ]based|0[- ]based)\))?"
DNA2 = r"((?:[ACGT](?:-(?=[ACGT]))?)+|(?:[acgt](?:-(?=[acgt]))?)+)"  # ACGT or A-C-G-T  # 1+ bases; a lowercase match whose length != the interval is treated as prose ("are at the very start")
INTERVAL = re.compile(
    r"(?i:bases|positions)\s+"
    + N
    + r"\s*(?:-|–|to|through)\s*"
    + N
    + CONV
    + r"\s+(?i:are|is|read|reads)\s+"
    + DNA2
    + r"\b"
)
LINE = re.compile(r"(?i:line starting at)\s+" + N + r"\s*\(([ACGTacgt-]+)\)")
ORDINAL = re.compile(
    r"(?i:position)\s+"
    + N
    + CONV
    + r"[^.;]{0,40}?(?i:is\s+the)\s+(\d{1,2}|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:st|nd|rd|th)?\s+(?i:of these|base of that line|base of the line|of them)"
)
POINT = re.compile(
    r"(?i:position|base)\s+" + N + CONV + r"\s+(?i:is|=)\s+(?:(?i:an?)\s+)?([ACGT])(?![A-Za-z])(?!\s+in\s+A\b)"
)
BSIDE = re.compile(r"(?i:position)\s+" + N + CONV + r"\s+(?i:is)\s+([ACGT])\s+in\s+A\s+and\s+([ACGT])\s+in\s+B")
CODON_IV = re.compile(r"(?i:codon)\s+" + N + r"\b[^.;]{0,60}?(?i:bases|positions)\s+" + N + r"\s*(?:-|–|to)\s*" + N)
ENCODES = re.compile(
    r"\b([ACGT]{3})\b,?\s+(?:(?i:which|that)\s+)?(?:(?i:also)\s+)?(?i:encodes|codes for|specifies)\s+(?:(?i:an?)\s+)?([A-Z*]|(?i:stop))\b"
)
HAS_AA = re.compile(r"\b([ACGT]{3})\s+\(([A-Z]|(?i:stop))\)")
KEEPS = re.compile(r"(?i:keeps codon)\s+" + N + r"\s+(?i:as)\s+([A-Z])\b")
ORDINAL_BASE = re.compile(
    r"(?i:the)\s+(\d{1,2}|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)(?:st|nd|rd|th)?\s+(?i:base)\s+(?i:is)\s+([ACGT])(?![A-Za-z])"
)
LENGTH = re.compile(r"(?i:\bthe sequence)\s+(?i:is|has)\s+" + N + r"\s*(?i:bases|bp|nt|nucleotides)\b")
ORF_LEN = re.compile(N + r"\s*(?:-|–)\s*" + N + r"\s*\(" + N + r"\s+(?i:bases|bp) including the stop codon\)")
CONTAIN = re.compile(r"(?i:bases)\s+" + N + r"\s*(?:-|–)\s*" + N + r"\s+(?i:contain)\s+" + N + r"\b")
DESIGN_GC = re.compile(N + r"\s+(?i:G or C)\s+\((\d{1,3}(?:\.\d{1,2})?)%\)")
OCCURS = re.compile(r"\b([ACGTacgt]{6,})\s+(?i:occurs)\s+(?:(?i:exactly)\s+)?(?i:(once|twice|(\d{1,4})\s+times))")


def _pos(num, tag, conv):
    n = to_int(num)
    c = tag or conv or "1-based"
    return n + 1 if c.startswith("0") else n


def _ord(x):
    return ORD.get(x.lower()) if not x.isdigit() else int(x)


ALT_CUE = re.compile(
    r"(?i:\b(?:after the (?:edit|change|substitution|deletion|insertion)|edited|mutated|mutant|the new sequence|the result|resulting|"
    r"in B\b|sequence B\b|the clone|clone's|once (?:changed|edited)))"
)
MAIN_CUE = re.compile(
    r"(?i:\b(?:before the (?:edit|change)|original|reference|in A\b|sequence A\b|unchanged|as given|as supplied))"
)  # \b: "unedited" is not "edited"


def claims_ok(reply, seqs, conv="1-based", gold=None):
    """(all typed claims true, positions covered by true interval / line claims of >= 5 bases).

    A positional claim is checked against the ORIGINAL sequence (seqs[0]) unless its sentence names an alternative
    ("after the edit", "in B", "the clone", "the mutated sequence"), in which case the alternatives (seqs[1:]) are
    used. Without this, a false claim about the original can be excused because it happens to hold in the edited
    sequence."""
    text = re.sub(r"<answer>.*?</answer>", " ", reply, flags=re.S)
    seqs = [s for s in seqs if s]
    main = seqs[0] if seqs else ""
    alts = seqs[1:]
    ok, covered = True, set()

    def clause_pool(sent, idx, default=None):
        """Pool for a claim at offset idx: the NEAREST cue before it decides which sequence it is about, so
        "in A bases 6-16 are ..., and in B bases 6-16 are ..." checks each half against the right sequence."""
        lo = max([sent.rfind(c, 0, idx) for c in ":;"] + [-1]) + 1  # a ':' or ';' ends the previous cue's scope
        last, which = -1, None
        for rx, w in ((ALT_CUE, "alt"), (MAIN_CUE, "main")):
            for m in rx.finditer(sent[lo:idx]):
                if m.start() > last:
                    last, which = m.start(), w
        return (alts or [main]) if which == "alt" else [main] if which == "main" else (default or [main])

    def in_any(a, b, s, pool):
        return any(1 <= a <= b <= len(q) and q[a - 1 : b] == s for q in pool)

    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        starts = []
        for m in INTERVAL.finditer(sent):
            a, b, raw = _pos(m.group(1), m.group(3), conv), _pos(m.group(2), m.group(3), conv), m.group(4)
            if raw.islower() and b - a + 1 != len(raw.replace("-", "")):
                continue  # English word after "bases A-B are", not a sequence claim
            s = raw.replace("-", "").upper()
            good = b - a + 1 == len(s) and in_any(a, b, s, clause_pool(sent, m.start()))
            ok &= good
            starts.append(a)
            if good and len(s) >= 2:
                covered |= set(range(a, b + 1))  # a 2+ base interval is real work; a 1-base point claim is not
        for m in LINE.finditer(sent):
            st, s = to_int(m.group(1)), m.group(2).replace("-", "").upper()
            good = (st - 1) % 60 == 0 and in_any(st, st + len(s) - 1, s, clause_pool(sent, m.start()))
            ok &= good
            starts.append(st)
            if good and len(s) >= 2:
                covered |= set(range(st, st + len(s)))
        for m in ORDINAL.finditer(sent):
            p, n = _pos(m.group(1), m.group(2), conv), _ord(m.group(3))
            ok &= bool(starts) and n is not None and any(p - s0 + 1 == n for s0 in starts)
        for m in BSIDE.finditer(sent):
            p = _pos(m.group(1), m.group(2), conv)
            a_, b_ = (gold or {}).get("a", main), (gold or {}).get("b", "")
            ok &= (
                1 <= p <= len(a_)
                and a_[p - 1] == m.group(3)
                and (not b_ or (p <= len(b_) and b_[p - 1] == m.group(4)))
            )
        for m in POINT.finditer(sent):
            p = _pos(m.group(1), m.group(2), conv)
            ok &= any(1 <= p <= len(q) and q[p - 1] == m.group(3) for q in clause_pool(sent, m.start()))
        for m in CODON_IV.finditer(sent):
            k, a = to_int(m.group(1)), to_int(m.group(2))
            ok &= (a - 1) % 3 == 0 and (a - 1) // 3 + 1 == k
        for m in ENCODES.finditer(sent):
            aa = "*" if m.group(2).lower() == "stop" else m.group(2)
            ok &= CODON[m.group(1)] == aa
        for m in HAS_AA.finditer(sent):
            aa = "*" if m.group(2).lower() == "stop" else m.group(2)
            ok &= CODON[m.group(1)] == aa
        for m in KEEPS.finditer(sent):
            k = to_int(m.group(1))
            prot = translate(main)
            ok &= 1 <= k <= len(prot) and prot[k - 1] == m.group(2)
        for m in ORDINAL_BASE.finditer(sent):
            if starts:
                continue  # "the 3rd base" inside a block sentence is handled by ORDINAL
            n = _ord(m.group(1))
            ok &= n is not None and any(
                1 <= n <= len(q) and q[n - 1] == m.group(2) for q in clause_pool(sent, m.start())
            )
        for m in LENGTH.finditer(sent):
            ok &= any(
                to_int(m.group(1)) == len(q) for q in clause_pool(sent, m.start(), seqs)
            )  # a cue before it names the sequence (an indel changes the length); else any
        for m in ORF_LEN.finditer(sent):
            ok &= to_int(m.group(2)) - to_int(m.group(1)) + 1 == to_int(m.group(3))
        for m in CONTAIN.finditer(sent):
            a, b, n = to_int(m.group(1)), to_int(m.group(2)), to_int(m.group(3))
            tgt = (gold or {}).get("count_target", "GC")
            ok &= 1 <= a <= b <= len(main) and sum(main[a - 1 : b].count(x) for x in tgt) == n
        for m in DESIGN_GC.finditer(sent):
            n, pct = to_int(m.group(1)), float(m.group(2))
            g = main.count("G") + main.count("C")
            ok &= n == g and bool(main) and abs(100 * g / len(main) - pct) <= 0.05 + 1e-9
        for m in OCCURS.finditer(sent):
            n = {"once": 1, "twice": 2}.get(m.group(2).lower()) or int(m.group(3))
            ok &= any(len(occurrences(q, m.group(1).upper())) == n for q in clause_pool(sent, m.start()))
    if gold and gold.get("task") in ("revstrand_pos", "genomic_pos"):
        allowed = {int(x) for x in gold.get("numbers", [])} | {0, 1, 3, 5}  # 1-based, 0-based, 5'/3' ends
        for x in re.findall(N, text):
            ok &= to_int(x) in allowed
    return ok, covered


# ---------------- per-task answer checkers ----------------
def chk_dna_exact(a, g):
    return as_dna(a) == g["answer"]


def chk_base(a, g):
    return len(a) == 1 and a.upper() == g["answer"]


def chk_flanks(a, g):
    parts = [p for p in re.split(r"[\s,;|/]+", a) if p]
    return (
        len(parts) == 2
        and all(re.fullmatch(r"[ACGTacgt]+", p) for p in parts)
        and [p.upper() for p in parts] == [g["up"], g["down"]]
    )


def chk_int(a, g):
    n = as_int(a)
    return n is not None and n == int(g["answer"])


def chk_check_base(a, g):
    a = a.lower()
    if g["answer"] == "yes":
        return a == "yes"
    m = re.fullmatch(r"no(?:\s*[,:]\s*|\s+)([acgt])", a)  # "not" is not "no, t"
    return bool(m) and m.group(1).upper() == g["actual"]


def chk_diff(a, g, single_snv=False):
    if g["answer"] == "none":
        return a.lower() == "none"
    ev = parse_events(a)
    if not ev or len(ev) != g.get("n_events", 1) or len({e[0] for e in ev}) != len(ev):
        return False
    if not all(minimal(e, single_snv) for e in ev):
        return False
    s = g["a"]
    for pos, ref, alt in sorted(ev, key=lambda e: -e[0]):  # right to left so coordinates stay on A
        s = apply_edit(s, pos, ref, alt)
        if s is None:
            return False
    return s == g["b"]


def chk_silent(a, g):
    if g["answer"] == "impossible":
        return a.lower() == "impossible"
    s, ref = as_dna(a), g["cds"]
    if s is None or len(s) != len(ref) or s == ref:
        return False
    diffs = [i for i in range(len(s)) if s[i] != ref[i]]
    lo = 3 * (g["codon"] - 1)
    return len(diffs) == 1 and lo <= diffs[0] < lo + 3 and translate(s) == translate(ref)


def chk_remove_site(a, g):
    s, ref, site = as_dna(a), g["cds"], g["site"]
    if s is None or len(s) != len(ref):
        return False
    if site in s or revcomp(site) in s:
        return False
    return translate(s) == translate(ref) and 0 < sum(x != y for x, y in zip(s, ref, strict=False)) <= 3


def chk_translate(a, g):
    a = a.upper()
    return re.fullmatch(r"[A-Z]+\*?", a) is not None and set(a.rstrip("*")) <= AMINO and a.rstrip("*") == g["answer"]


def chk_first_stop(a, g):
    if g["answer"] == "none":
        return a.lower() == "none"
    m = re.fullmatch(N + r"\s+(TAA|TAG|TGA)", a.upper())
    return bool(m) and to_int(m.group(1)) == g["pos"] and m.group(2) == g["codon_seq"]


ITEM_STRANDED = re.compile(N + r"\s*\(?([+-])\)?")


def chk_positions_set(a, g):
    """'none' or the exact set of 1-based positions (with strand for PAMs); separators ';', ', ' or whitespace."""
    if not g["answer_set"]:
        return a.lower() == "none"
    if g.get("stranded"):
        items = ITEM_STRANDED.findall(a)
        if re.sub(r"[;,\s]", "", ITEM_STRANDED.sub("", a)):
            return False
        got = [(to_int(p), s) for p, s in items]
    else:
        toks = [t for t in re.split(r";|,\s+|\s+", a.strip()) if t]
        if not all(re.fullmatch(N, t) for t in toks):
            return False
        got = [to_int(t) for t in toks]
    want = {tuple(x) if isinstance(x, list) else x for x in g["answer_set"]}
    return set(got) == want and len(got) == len(set(got))


def chk_token(a, g):
    a = a.lower()
    want = g["answer"].lower()
    return a == want or (want == "iupac dna" and a == "iupac")


def chk_count(a, g):
    if "percent" in g:  # full format required: COUNT/TOTAL (PERCENT%)
        m = re.fullmatch(N + r"\s*/\s*" + N + r"\s*\(\s*(\d{1,3}(?:\.\d{1,2})?)\s*%\s*\)", a)
        if not m or to_int(m.group(1)) != g["count"] or to_int(m.group(2)) != g["total"]:
            return False
        if not re.fullmatch(r"\d{1,3}\.\d", m.group(3)):
            return False  # one decimal, as the prompt asks
        return (
            abs(float(m.group(3)) - 100 * g["count"] / g["total"]) <= 0.05 + 1e-9
        )  # either rounding of a .x5 tie is accepted
    n = as_int(a)
    return n is not None and n == g["count"]


def max_repeat_run(s, k=8):
    """Longest run of consecutive copies of one k-mer (degeneration signal: ACTCCGCCCG x 580)."""
    best = 0
    for i in range(len(s) - k + 1):
        u = s[i : i + k]
        n = 1
        while s[i + n * k : i + (n + 1) * k] == u:
            n += 1
        best = max(best, n)
    return best


def chk_design(a, g):
    s = as_dna(a)
    if s is None:
        return False
    c = g["constraints"]
    if len(s) != c["length"]:
        return False
    if c.get("max_kmer_runs") and max_repeat_run(s) > c["max_kmer_runs"]:
        return False  # no degenerate tandem repeats
    gc = 100 * (s.count("G") + s.count("C")) / len(s)
    if not (c["gc_min"] - 1e-9 <= gc <= c["gc_max"] + 1e-9):
        return False
    if any(f in s or revcomp(f) in s for f in c.get("forbidden", [])):
        return False
    if any(len(occurrences(s, r)) != 1 for r in c.get("required_once", [])):
        return False
    return s.startswith(c.get("prefix", "")) and s.endswith(c.get("suffix", ""))


def chk_orf(a, g):
    if g["answer"] == "none":
        return a.lower() == "none"
    m = re.fullmatch(N + r"\s*[-–]\s*" + N + r"\s*\(?([+-])\)?", a)
    return bool(m) and (to_int(m.group(1)), to_int(m.group(2)), m.group(3)) in {tuple(x) for x in g["answer_set"]}


CHECKERS = {
    "base_at": chk_base,
    "span_at": chk_dna_exact,
    "after_snippet": chk_dna_exact,
    "apply_edit": chk_dna_exact,
    "indel_edit": chk_dna_exact,
    "gibson": chk_dna_exact,
    "revcomp": chk_dna_exact,
    "flanks": chk_flanks,
    "snippet_pos": chk_int,
    "revstrand_pos": chk_int,
    "genomic_pos": chk_int,
    "check_base": chk_check_base,
    "diff": chk_diff,
    "multi_snv": lambda a, g: chk_diff(a, g, single_snv=True),
    "silent_edit": chk_silent,
    "remove_site": chk_remove_site,
    "translate": chk_translate,
    "first_stop": chk_first_stop,
    "motif_sites": chk_positions_set,
    "pam_sites": chk_positions_set,
    "input_type": chk_token,
    "clone_compare": chk_token,
    "start_stop": chk_token,
    "count": chk_count,
    "design": chk_design,
    "longest_orf": chk_orf,
}


def answer_ok(ans, gold):
    a = unwrap(ans)
    if gold.get("answer") == "out of range":
        return a.lower() == "out of range"
    if gold["task"] == "pushback":
        return answer_ok(ans, gold["inner"])
    return CHECKERS[gold["task"]](a, gold)


def task_seqs(g, answer):
    if g["task"] == "design":
        return [as_dna(unwrap(answer)) or ""]
    main = g.get("seq") or g.get("a") or g.get("cds") or ""
    return (
        [main]
        + [g[k] for k in ("b", "edited", "clone") if g.get(k)]
        + ([as_dna(unwrap(answer))] if g["task"] in ("silent_edit", "remove_site") and as_dna(unwrap(answer)) else [])
    )


def _verify(reply, gold, mode):
    if not isinstance(reply, str) or len(reply) > MAX_REPLY:
        return 0.0
    answers = final_answers(reply)
    if not answers or len(answers) != 1:
        return 0.0
    if not answer_ok(answers[0], gold):
        return 0.0
    if mode == "strict":
        g = gold["inner"] if gold["task"] == "pushback" else gold
        seqs = task_seqs(g, answers[0])
        if seqs and seqs[0]:
            ok, covered = claims_ok(reply, seqs, g.get("conv", "1-based"), g)
            if not ok:
                return 0.0
            if g["task"] in ("base_at", "check_base") and g.get("answer") != "out of range" and g["x1"] not in covered:
                return 0.0
    return 1.0


def verify(reply, gold_answer_json, mode="strict"):
    try:
        gold = json.loads(gold_answer_json) if isinstance(gold_answer_json, str) else gold_answer_json
        return _verify(reply, gold, mode)
    except Exception:
        return 0.0  # malformed input never crashes the trainer
