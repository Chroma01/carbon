"""Optional <dna>...</dna> tagging for evaluating hybrid (6mer/char-tail) tokenizer checkpoints.

The hybrid midtraining data was tagged with the SAME tagger (tag_dna_sequences), so eval prompts
must match: A/C/G/T runs >=6 get wrapped in <dna>...</dna> and routed through the 6mer + tail path
instead of BPE. Gate behind --tag-dna so BPE checkpoints (e.g. abl21) stay UNtagged (their tokenizer
has no <dna> token; tagging would inject literal '<','dna','>' text tokens).

Note on BPB: tagging changes how the answer is tokenized but NOT its content, so always normalize
bits-per-byte by the ORIGINAL (untagged) answer's UTF-8 byte length.
"""
import sys, os

_TAGGER = None
_TAGGER_PATH = os.path.dirname(os.path.abspath(__file__))


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        if _TAGGER_PATH not in sys.path:
            sys.path.insert(0, _TAGGER_PATH)
        from tag_dna_sequences import tag_dna_sequences
        _TAGGER = tag_dna_sequences
    return _TAGGER


def tag(text, enabled):
    """Return text with DNA runs wrapped in <dna>..</dna> if enabled, else text unchanged."""
    if not enabled or not text:
        return text
    return _tagger()(str(text))[0]
