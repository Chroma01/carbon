import re
from typing import List, Tuple, Dict, Any, Optional

DNA_TOKEN_RE = re.compile(r"^[ATCGatcg]{6,}$")
DNA_IUPAC_RE = re.compile(r"^[ATCGNRYKMSWBDHVatcgnrykmswbdhv]{6,}$")
# For probe/primer strings like 5′-6FAM-ATCG...-MGB-NFQ-3′, split only on dashes.
DASH_SPLIT_RE = re.compile(r"([-–—])")
# Leading/trailing punctuation/markers commonly seen around sequences:
# includes slashes, backslashes, primes (ASCII + unicode), digits (e.g. 5' / 3'),
# hyphens and unicode dashes.
PUNCT_EDGE_RE = re.compile(r"^([\\/\(\[\{\"'`′0-9\-–—]*)(.*?)([\\/\)\]\}\"'`′0-9,\;:\.\!\?\-–—]*)$")


def tag_dna_sequences(
    text: str,
    min_len: int = 6,
    max_ambig_frac: float = 0.2,
) -> Tuple[str, List[int], List[str]]:
    """
    Tag DNA-like tokens in text with <dna>...</dna>.
    Criteria:
    - Token must be a single whitespace-delimited word (after stripping edge punctuation)
    - Token must be only A/T/C/G (case-insensitive) OR IUPAC codes (len >= 12) with ambiguity <= max_ambig_frac
    - Token length >= min_len
    Returns:
        tagged_text, dna_word_ids, dna_sequences
        dna_word_ids are indices among non-whitespace tokens (0-based)
    """
    if not text:
        return text, [], []

    # Preserve original whitespace by splitting into tokens + separators
    parts = re.split(r"(\s+)", text)

    dna_word_ids: List[int] = []
    dna_sequences: List[str] = []

    word_index = 0
    for i, part in enumerate(parts):
        if not part or part.isspace():
            continue

        # Strip leading/trailing punctuation for matching only
        lead, core, trail = part, part, ""
        m = PUNCT_EDGE_RE.match(part)
        if m:
            lead, core, trail = m.group(1), m.group(2), m.group(3)

        if core and len(core) >= min_len:
            tagged_any = False

            if DNA_TOKEN_RE.match(core):
                core_up = core.upper()
                parts[i] = f"{lead}<dna>{core_up}</dna>{trail}"
                dna_sequences.append(core_up)
                tagged_any = True
            elif DNA_IUPAC_RE.match(core):
                core_up = core.upper()
                ambig_count = sum(1 for c in core_up if c not in {"A", "T", "C", "G"})
                ambig_frac = ambig_count / len(core_up)
                if len(core_up) >= 12 and ambig_frac <= max_ambig_frac:
                    parts[i] = f"{lead}<dna>{core_up}</dna>{trail}"
                    dna_sequences.append(core_up)
                    tagged_any = True
            elif DASH_SPLIT_RE.search(core):
                subparts = DASH_SPLIT_RE.split(core)
                rebuilt = []
                for sp in subparts:
                    if not sp:
                        continue
                    if DASH_SPLIT_RE.fullmatch(sp):
                        rebuilt.append(sp)
                        continue
                    if len(sp) < min_len:
                        rebuilt.append(sp)
                        continue
                    if DNA_TOKEN_RE.match(sp):
                        sp_up = sp.upper()
                        rebuilt.append(f"<dna>{sp_up}</dna>")
                        dna_sequences.append(sp_up)
                        tagged_any = True
                    elif DNA_IUPAC_RE.match(sp):
                        sp_up = sp.upper()
                        ambig_count = sum(1 for c in sp_up if c not in {"A", "T", "C", "G"})
                        ambig_frac = ambig_count / len(sp_up)
                        if len(sp_up) >= 12 and ambig_frac <= max_ambig_frac:
                            rebuilt.append(f"<dna>{sp_up}</dna>")
                            dna_sequences.append(sp_up)
                            tagged_any = True
                        else:
                            rebuilt.append(sp)
                    else:
                        rebuilt.append(sp)
                if tagged_any:
                    parts[i] = f"{lead}{''.join(rebuilt)}{trail}"

            if tagged_any:
                dna_word_ids.append(word_index)

        word_index += 1

    return "".join(parts), dna_word_ids, dna_sequences


def tag_dna_sequences_batch(
    batch: Dict[str, List[str]],
    text_key: str = "text",
    min_len: int = 6,
    max_ambig_frac: float = 0.2,
) -> Dict[str, Any]:
    texts = batch[text_key]
    tagged_texts = []
    all_ids = []
    all_seqs = []

    for text in texts:
        tagged, ids, seqs = tag_dna_sequences(
            text,
            min_len=min_len,
            max_ambig_frac=max_ambig_frac,
        )
        tagged_texts.append(tagged)
        all_ids.append(ids)
        all_seqs.append(seqs)

    return {
        f"{text_key}_tagged": tagged_texts,
        "dna_word_ids": all_ids,
        "dna_sequences": all_seqs,
    }


if __name__ == "__main__":
    # Quick sanity tests (local, no dataset)
    examples = [
        "This is ATCGAT and ATCGATCG with words.",
        "codon ATG should not be tagged if min_len=6.",
        "lowercase atcgat should be tagged.",
        r"T\. miR\148\F: \CAAGCACGAT TAGCATTTGA\; miR\148\R: \TAGAAAGCT TTCGAGACAA\.",
        "5'ATCGATCG3' and 5′ATCGATCG3′ should be tagged.",
        "Punctuation ATCGAT, should be tagged.",
        "Probe 5′-6FAM-ATCGATCG-3′ should tag the DNA part.",
        "Mix: ATCGATCG ATCGATx ATCGATCGAT",
        "Multiple   spaces ATCGAT    and newlines\nATCGATCG",
        "IUPAC ATCGNNRY should NOT be tagged at max_ambig_frac=0.2.",
    ]

    for ex in examples:
        tagged, ids, seqs = tag_dna_sequences(ex, min_len=6)
        print("---")
        print("IN: ", ex)
        print("OUT:", tagged)
        print("IDS:", ids)
        print("SEQ:", seqs)