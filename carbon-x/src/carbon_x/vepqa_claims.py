# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Verify the factual claims in VEPQA templated reasoning (HuggingFaceBio/vepqa-sft-draft-loubna, config
`templated_reasoning`), for the `vepqa_groundedness` reward.

`verify(prompt, completion)` recomputes the truth from the prompt (reference DNA, 1-based position, ref/alt alleles)
and checks every sentence of the completion, with `<dna>` tags removed. A leading empty `<think></think>` block and a
terminal answer holding just the label, `<answer>benign</answer>` or `<answer>pathogenic</answer>` as `vepqa_accuracy`
scores it, are exempt. Any other think or answer tag is checked like the rest, so it cannot hide a claim.

A stated likelihood level is the rationale's own claim and is never compared with the row's stored `surprise_bin`: that
whole-window likelihood depends on the alternate sequence, which a single-view prompt never shows. Instead, every
likelihood statement must state the same level, its comparison and interpretation phrases must belong to that level,
and every concessive verdict (`CONCESSIONS`) must be consistent with it.

The supported grammar is the template's own sentences (`SENTENCES`), whose variable parts (bases, DNA strings, lengths,
level phrases) are checked. A sentence outside it is accepted only when it is connective or verdict wording made of
`CONNECTIVE_WORDS`; anything else fails closed, including negated or reworded claims, and annotation claims (gene,
protein, coding, conservation, database, splicing consequence) count as false claims. Required: the substitution and
its transition/transversion type, and the variant's 6-base block. Optional facts, including the likelihood level, are
checked whenever stated. Free prose (e.g. Gemini-written rationales) is out of scope and fails closed.

Definitions used to generate the data (s = reference window, i = position - 1):
- trinucleotide s[i-1:i+2] (clipped at the window edge); CpG site: ref C with next G, or ref G with previous C (i > 0)
- homopolymer: run of s[i] containing i (stated if >= 4); simple repeat: unit of 2-6 bases, >= 3 copies, >= 12 bp,
  covering i
- local GC: % G+C in s[max(0, i-50) : i+51], over its clipped length; AT-rich < 40 <= moderate GC < 50 <= moderately
  GC-rich < 60 <= GC-rich
- GT/AG lost or gained at (i-1, i) and (i, i+1)
- 6-base block s[6*(i//6) : +6] (one Carbon DNA token, clipped at the window edge but still called a 6-base block);
  flanks of 0/12/24/48 bases (40/20/20/20% of rows), shorter near the edge. A quoted flank must be the bases
  immediately on its stated side of the block, and a stated flank length must match the quoted bases.
- surprise_bin: LLR = log P(window with alt) - log P(window with ref), natural log, over the whole 8,192-bp window
  under Carbon-X-Base (HuggingFaceBio/Carbon-X-Base), binned at -1, -4.7, -9.0 and -17.5 into levels 0
  (above -1) to 4 (below -17.5)
"""

import re


REQUIRED = ("substitution", "mutation_type", "block")
GC_WORDS = {
    "AT-rich": (0, 40),
    "moderate GC": (40, 50),
    "moderately GC-rich": (50, 60),
    "GC-rich": (60, 101),
}  # % GC in s[i-50 : i+51], clipped
# The likelihood sentence is "<frame>, a <alt> at this position is <level> the reference <ref>, <comparison>,
# <interpretation>." Each level (indexed by `surprise_bin`) has its own comparisons and interpretations.
FRAMES = (
    "Based on typical genomic sequence patterns",
    "Given the sequence patterns typical of genomes",
    "Judging from how genomic DNA usually looks around such a site",
    "Compared with what the surrounding genomic sequence predicts",
)
LEVELS = (
    "about as likely as",
    "somewhat less likely than",
    "less likely than",
    "much less likely than",
    "far less likely than",
)
COMPARISONS = (
    ("or more",),
    (
        "a modest drop relative to other variants",
        "a smaller drop than many variants show",
        "a small drop compared with other variants",
    ),
    (
        "a mid-range drop among variants",
        "a moderate drop compared with other variants",
        "an intermediate drop relative to other variants",
    ),
    (
        "larger than the drop most variants produce",
        "a larger drop than most variants show",
        "a bigger drop than most variants cause",
    ),
    (
        "near the top of the range seen for variants",
        "one of the largest drops seen for variants",
        "among the largest drops seen for variants",
    ),
)
TOLERANT = (
    "which suggests the position tolerates change",
    "suggesting the site is not under strong constraint",
    "consistent with a position that tolerates variation",
)
INTERPRETATIONS = (
    TOLERANT,
    TOLERANT,
    (
        "consistent with a lightly constrained position",
        "pointing to weak constraint at this site",
        "which suggests little constraint",
    ),
    (
        "which suggests a constrained position",
        "consistent with a position under constraint",
        "pointing to a constrained site",
    ),
    (
        "a strong sign that this position is constrained",
        "which suggests a strongly constrained position",
        "pointing to a strongly constrained site",
    ),
)
# Verdicts that concede the likelihood evidence, and the levels each concession is consistent with
CONCESSIONS = {
    "Although the alternate base is not strongly disfavored here": {0, 1, 2},
    "Even though the drop in likelihood is not large": {0, 1, 2},
    "Although the alternate base is somewhat unexpected here": {1, 2, 3, 4},
    "Despite the drop in likelihood": {1, 2, 3, 4},
}


def _any(options):
    return "|".join(re.escape(option) for option in options)


# Every supported sentence, as (kind, pattern). A sentence must match one pattern in full.
SENTENCES = [
    (
        "locate",
        re.compile(
            r"Looking at the sequence around the variant, the (?P<n_up>\d+) bases (?P<up>[ACGT]+) come right before "
            r"the six-base block (?P<block>[ACGT]+), followed by (?P<down>[ACGT]+); the change turns the block into "
            r"(?P<block_alt>[ACGT]+)\."
        ),
    ),
    (
        "locate",
        re.compile(
            r"The variant falls in the (?P<n_block>\d+)-base block (?P<block>[ACGT]+), which becomes "
            r"(?P<block_alt>[ACGT]+)(?:; the (?P<n_up>\d+) bases before this block are (?P<up>[ACGT]+) and the "
            r"(?P<n_down>\d+) bases after it are (?P<down>[ACGT]+))?\."
        ),
    ),
    (
        "locate",
        re.compile(
            r"Locating the variant: it sits in the block (?P<block>[ACGT]+), which changes to (?P<block_alt>[ACGT]+)"
            r"(?:, with (?P<up>[ACGT]+) immediately upstream and (?P<down>[ACGT]+) downstream)?\."
        ),
    ),
    (
        "substitution",
        re.compile(
            r"At the variant position, (?P<ref>[ACGT]) is replaced by (?P<alt>[ACGT]), a "
            r"(?P<type>transition|transversion) that turns (?P<tri_ref>[ACGT]+) into (?P<tri_alt>[ACGT]+)\."
        ),
    ),
    (
        "substitution",
        re.compile(
            r"The variant changes (?P<ref>[ACGT]) to (?P<alt>[ACGT]) \(a (?P<type>transition|transversion)\), so "
            r"(?P<tri_ref>[ACGT]+) becomes (?P<tri_alt>[ACGT]+)\."
        ),
    ),
    (
        "substitution",
        re.compile(
            r"This is an? (?P<ref>[ACGT])>(?P<alt>[ACGT]) (?P<type>transition|transversion), changing the local "
            r"context from (?P<tri_ref>[ACGT]+) to (?P<tri_alt>[ACGT]+)\."
        ),
    ),
    (
        "cpg_site",
        re.compile(
            r"The site is a CpG, where changes of this kind arise often\.|"
            r"The reference base sits in a CpG dinucleotide, a mutation-prone context\."
        ),
    ),
    (
        "creates_cpg",
        re.compile(
            r"The alternate base forms a new CpG with its neighbour\.|The change creates a new CpG dinucleotide\."
        ),
    ),
    ("homopolymer", re.compile(r"It falls in a run of (?P<length>\d+) (?P<base>[ACGT]) bases\.")),
    ("homopolymer", re.compile(r"The site is part of an? (?P<base>[ACGT])-homopolymer of (?P<length>\d+) bases\.")),
    ("homopolymer", re.compile(r"It sits in a stretch of (?P<length>\d+) consecutive (?P<base>[ACGT])s\.")),
    ("repeat", re.compile(r"It lies inside a \((?P<unit>[ACGT]+)\)n simple repeat\.")),
    ("repeat", re.compile(r"It falls in a short repeated \((?P<unit>[ACGT]+)\)n stretch\.")),
    ("repeat", re.compile(r"The site sits within a \((?P<unit>[ACGT]+)\)n tandem repeat\.")),
    (
        "local_gc",
        re.compile(
            rf"(?:Around the site, the sequence is|The surrounding ~100 bases are|The local context is) "
            rf"(?P<gc>{_any(GC_WORDS)})\."
        ),
    ),
    (
        "gt_ag",
        re.compile(
            r"The change (?P<verb>creates a new|removes an?) (?P<dinucleotide>GT|AG) dinucleotide, the kind found at "
            r"splice sites, though such dinucleotides are common\."
        ),
    ),
    (
        "likelihood",
        re.compile(
            rf"(?:{_any(FRAMES)}), an? (?P<alt>[ACGT]) at this position is (?P<level>{_any(LEVELS)}) the reference "
            rf"(?P<ref>[ACGT]), (?P<comparison>{_any(c for level in COMPARISONS for c in level)}), "
            rf"(?P<interpretation>{_any(dict.fromkeys(i for level in INTERPRETATIONS for i in level))})\."
        ),
    ),
    (
        "concession",
        re.compile(
            rf"(?P<concession>{_any(CONCESSIONS)}), (?:the overall evidence still points to a (?:benign|pathogenic) "
            rf"variant|the variant is still most likely (?:benign|pathogenic))\."
        ),
    ),
]
# Connective and verdict wording, e.g. "Overall, the variant is most likely benign." A sentence made only of these words
# (and basic punctuation) carries no checkable claim.
CONNECTIVE_WORDS = frozenset(
    "a an and answer as balance be benign best by classified conclusion evidence final given i I in is it let let's "
    "likely me most of on overall pathogenic points probably so step still summary taken the therefore think this thus "
    "to together variant".split()
)
# Claims the prompt cannot support: it holds only a DNA window, a position and the two alleles. A GT/AG dinucleotide is
# checkable, but a splice-site consequence is an annotation claim. The GT/AG sentence's own hedge is not one, even in a
# reworded sentence, which is unsupported wording instead.
SPLICE_HEDGE = "the kind found at splice sites, though such dinucleotides are common"
ANNOTATION = re.compile(
    r"\b(genes?|genic|protein\w*|amino[- ]acids?|codons?|exons?|introns?|intronic|exonic|coding|non-?coding|"
    r"conserv\w*|clinvar|gnomad|databases?|missense|nonsense|synonymous|frameshift|transcripts?|annotat\w*|splic\w*|"
    r"donors?|acceptors?|UTRs?|promoters?|enhancers?|regulatory|domains?|function\w*)\b",
    re.I,
)


def parse_prompt(prompt):
    seqs = re.findall(r"[ACGTN]{1000,}", prompt.replace("<dna>", "").replace("</dna>", ""))
    pos = int(re.search(r"position (\d+)", prompt).group(1))
    m = re.search(r"reference allele ([ACGT]) is changed to the alternate allele ([ACGT])", prompt)
    return seqs[0].upper(), pos - 1, m.group(1), m.group(2)


def truth(s, i, ref, alt):
    assert s[i] == ref, "reference allele does not match the sequence"
    b0 = 6 * (i // 6)
    a = s[:i] + alt + s[i + 1 :]
    lo, hi = i, i
    while lo > 0 and s[lo - 1] == s[i]:
        lo -= 1
    while hi + 1 < len(s) and s[hi + 1] == s[i]:
        hi += 1
    w = s[max(0, i - 50) : i + 51]

    def di(x):
        return {x[j : j + 2] for j in (i - 1, i) if j >= 0 and j + 2 <= len(x)}

    return {
        "ref": ref,
        "alt": alt,
        "transition": {ref, alt} in ({"A", "G"}, {"C", "T"}),
        "tri": (s[max(0, i - 1) : i + 2], a[max(0, i - 1) : i + 2]),
        "block": s[b0 : b0 + 6],
        "block_alt": a[b0 : b0 + 6],
        "b0": b0,
        "cpg_site": (ref == "C" and s[i + 1 : i + 2] == "G") or (ref == "G" and i > 0 and s[i - 1] == "C"),
        "creates_cpg": (alt == "C" and s[i + 1 : i + 2] == "G") or (alt == "G" and i > 0 and s[i - 1] == "C"),
        "homo": (s[i], hi - lo + 1),
        "gc": 100 * (w.count("G") + w.count("C")) / len(w),
        "gtag_lost": {d for d in di(s) if d in ("GT", "AG")} - di(a),
        "gtag_gained": {d for d in di(a) if d in ("GT", "AG")} - di(s),
    }


def verify(prompt: str, completion: str) -> dict:
    """
    Check every statement of a templated VEPQA rationale against its prompt.

    Args:
        prompt (`str`):
            User text of the VEPQA question, with the reference DNA window (plain or in `<dna>` tags), the 1-based
            variant position and the two alleles.
        completion (`str`):
            Model completion or reference answer. `<dna>` tags are removed, a leading empty `<think></think>` block is
            ignored, and only a terminal answer holding just the label is ignored.

    Returns:
        `dict` with keys:
            - `claims` (`dict[str, bool]`):
                For each stated claim family, whether all of its statements are true. Families that are not stated are
                absent.
            - `missing` (`list[str]`):
                Required families that are never stated.
            - `unsupported` (`list[str]`):
                Sentences outside the supported grammar that are not connective or verdict wording.
            - `grounded` (`bool`):
                Whether nothing is missing or unsupported and every claim is true.
    """
    s, i, ref, alt = parse_prompt(prompt)
    T = truth(s, i, ref, alt)
    b0 = T["b0"]
    claims, stated, unsupported = {}, set(), []
    levels, concessions = set(), []

    def check(family, ok):
        claims[family] = claims.get(family, True) and bool(ok)

    # Generation can prepend an empty thinking block. Only one empty block at the start is exempt; any non-empty,
    # repeated or later think tag fails closed below.
    body = re.sub(r"\A\s*<think>\s*</think>\s*", "", completion, count=1, flags=re.I)
    # Only the terminal label that `vepqa_accuracy` scores is exempt; any other answer tag fails closed below
    body = re.sub(r"<answer>\s*(?:benign|pathogenic)\s*</answer>\s*\Z", "", body, flags=re.I)
    body = re.sub(r"</?dna>", "", body)
    for sentence in re.split(r"(?<=[.!?])\s*|\n+", body):
        sentence = sentence.strip()
        if not sentence:
            continue
        kind, g = None, {}
        for name, pattern in SENTENCES:
            m = pattern.fullmatch(sentence)
            if m:
                kind, g = name, m.groupdict()
                break
        if kind == "locate":
            stated.add("block")
            check("block", g["block"] == T["block"] and int(g.get("n_block") or 6) == 6)
            check("mutated_block", g["block_alt"] == T["block_alt"])
            # Each flank must be on its stated side of the block: the bases right before it, or right after it
            if g["up"] is not None:
                up = g["up"]
                check("flanks", s[max(0, b0 - len(up)) : b0] == up and int(g.get("n_up") or len(up)) == len(up))
            if g["down"] is not None:
                down = g["down"]
                check(
                    "flanks", s[b0 + 6 : b0 + 6 + len(down)] == down and int(g.get("n_down") or len(down)) == len(down)
                )
        elif kind == "substitution":
            stated.update({"substitution", "mutation_type"})
            check("substitution", (g["ref"], g["alt"]) == (ref, alt))
            check("mutation_type", (g["type"] == "transition") == T["transition"])
            check("trinucleotide", (g["tri_ref"], g["tri_alt"]) == T["tri"])
        elif kind == "cpg_site":
            check("cpg", T["cpg_site"])
        elif kind == "creates_cpg":
            check("cpg", T["creates_cpg"])
        elif kind == "homopolymer":
            check("homopolymer", (g["base"], int(g["length"])) == T["homo"])
        elif kind == "repeat":
            u = g["unit"]
            # Repeats from every start, so a greedy match ending before i cannot hide one covering i, e.g. (ATAT)n in
            # the last 12 bases of a 14-base AT run
            check(
                "repeat",
                2 <= len(u) <= 6
                and any(
                    x.start() <= i < x.start() + len(x.group(1)) and len(x.group(1)) >= 12
                    for x in re.finditer(f"(?=((?:{u}){{3,}}))", s)
                ),
            )
        elif kind == "local_gc":
            lo, hi = GC_WORDS[g["gc"]]
            check("local_gc", lo <= T["gc"] < hi)
        elif kind == "gt_ag":
            check(
                "gt_ag", g["dinucleotide"] in (T["gtag_lost"] if g["verb"].startswith("removes") else T["gtag_gained"])
            )
        elif kind == "likelihood":
            level = LEVELS.index(g["level"])
            levels.add(level)
            check(
                "likelihood", g["comparison"] in COMPARISONS[level] and g["interpretation"] in INTERPRETATIONS[level]
            )
            # The likelihood sentence restates both alleles
            check("substitution", (g["ref"], g["alt"]) == (ref, alt))
        elif kind == "concession":
            concessions.append(CONCESSIONS[g["concession"]])
        elif ANNOTATION.search(sentence.replace(SPLICE_HEDGE, "")):
            check("annotation", False)
        elif re.search(r"[^A-Za-z' ,;:.!?]", sentence) or not all(
            (word.lower() if index == 0 else word) in CONNECTIVE_WORDS
            for index, word in enumerate(re.findall(r"[A-Za-z']+", sentence))
        ):
            unsupported.append(sentence)
    # Every likelihood statement must state the same level, and every concession must be consistent with it
    if levels or concessions:
        check("likelihood", len(levels) <= 1 and all(levels <= allowed for allowed in concessions))
    missing = [family for family in REQUIRED if family not in stated]
    return {
        "claims": claims,
        "missing": missing,
        "unsupported": unsupported,
        "grounded": not missing and not unsupported and all(claims.values()),
    }
