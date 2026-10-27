"""
HybridDNATokenizer: Combines Qwen3 BPE tokenization with DNA 6-mer tokenization.

Vendored from HuggingFaceBio/Carbon-X-Base at
c8122fbb1f90c39e325de57500ae027d070dec96. Local changes preserve effective
EOS/padding tokens when saving and reloading SFT checkpoints.

DNA spans are detected automatically or supplied in <dna>...</dna> tags and
tokenized as 6-mers. All remaining text uses Qwen3's BPE tokenization.

Supports token_mask for Fine-grained Nucleotide Supervision (FNS):
  -2: padding token
  -1: text token (BPE)
   0: DNA special token (<dna>, </dna>, <oov>)
  1-5: partial 6-mer token — valid_length real bases at positions [0, valid_length),
       right-padded with 'A' at positions [valid_length, k) so loss can supervise
       positions 0..valid_len-1 via pos_mask = (valid_len > pos)
   6: full 6-mer
"""

import os
import json
import itertools
import re
from typing import List, Optional, Tuple, Dict, Union, Any

from transformers import PreTrainedTokenizer, AutoTokenizer, BatchEncoding

# The BPE vocabulary comes from this tokenizer, and its size sets where the DNA ids start. Pin it so that a change on the
# Hub cannot silently shift the ids that checkpoints were trained with.
BASE_TOKENIZER = "Qwen/Qwen3-4B-Base"
BASE_TOKENIZER_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


class HybridDNATokenizer(PreTrainedTokenizer):
    """
    Hybrid tokenizer combining Qwen3 BPE with DNA 6-mer tokenization.

    Explicit <dna>...</dna> regions are always tokenized as 6-mers. By default,
    untagged DNA spans embedded in text are detected and tagged automatically;
    pass auto_dna_tags=False to encode() or __call__() to disable detection for
    one call.
    """

    model_input_names = ["input_ids", "attention_mask"]

    # Single-nucleotide alphabet for tail="char": standard bases (A C T G U) +
    # IUPAC ambiguity codes (R Y S W K M B D H V N). These 16 chars match
    # microsoft/NatureLM-8x7B nlm_tokenizer.py's single-char DNA set and the
    # NatureLM paper's stated DNA vocabulary size of 16 (arXiv:2502.07527, S2.1).
    SINGLE_BASE_ALPHABET = "ACTGURYSWKMBDHVN"
    IUPAC_DNA_ALPHABET = "ACGTRYSWKMBDHVN"
    STANDARD_DNA_ALPHABET = frozenset("ACGT")
    IUPAC_RUN_RE = re.compile(f"[{IUPAC_DNA_ALPHABET}]+", re.IGNORECASE | re.ASCII)
    SPACED_DNA_RE = re.compile(
        r"(?<![A-Za-z])(?:[ACGT]{1,5}[ \t]+){5,}[ACGT]{1,5}(?![A-Za-z])"
    )

    def __init__(
        self,
        base_tokenizer_path: Optional[str] = None,
        k: int = 6,
        auto_dna_tags: bool = True,
        tail: str = "pad_a",
        **kwargs
    ):
        if tail not in ("pad_a", "pad_n", "char"):
            raise ValueError(f"tail must be 'pad_a', 'pad_n' or 'char', got {tail!r}")
        self.tail = tail
        self.pad_char = {"pad_a": "A", "pad_n": "N"}.get(tail)
        self.k = k

        # Load base tokenizer (Qwen3-4B-Base) at the pinned revision
        self._base_tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER, revision=BASE_TOKENIZER_REVISION)

        # Get base vocabulary
        self._base_vocab = self._base_tokenizer.get_vocab()
        self._base_vocab_size = len(self._base_vocab)

        # Initialize DNA vocabulary
        self._init_dna_vocab()

        # Build combined vocabulary
        self._build_combined_vocab()

        # Set special tokens
        self._eos_token = kwargs.pop('eos_token', None) or self._base_tokenizer.eos_token or "<|endoftext|>"
        self._pad_token = kwargs.pop('pad_token', None) or self._base_tokenizer.pad_token or "<|endoftext|>"

        # Initialize parent class
        super().__init__(
            eos_token=self._eos_token,
            pad_token=self._pad_token,
            **kwargs
        )

        # Preserve the base tokenizer's AddedToken metadata. These tokens are
        # already present in the combined vocabulary at their original IDs, so
        # this only registers their metadata and does not grow the vocabulary.
        self.add_tokens(list(self._base_tokenizer.added_tokens_decoder.values()))

        self.special_tokens = self.dna_special_tokens + [self._eos_token, self._pad_token]
        self.auto_dna_tags = auto_dna_tags

    def _init_dna_vocab(self):
        """Initialize DNA vocabulary (special tokens + k-mers + padding for 128 alignment)."""
        bases = ['A', 'T', 'C', 'G']

        # DNA special tokens
        self.dna_special_tokens = ["<dna>", "</dna>", "<oov>"]

        # Generate all k-mer combinations (4^k = 4096 for k=6)
        self.kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=self.k)]

        # tail="pad_n": dedicated tokens for the N-padded tail ("CT" -> "CTNNNN"),
        # one per real prefix of length 1..k-1, i.e. 4^1+...+4^(k-1) = 1364.
        self.npad_tokens = []
        if self.tail == "pad_n":
            for prefix_len in range(1, self.k):
                for prefix in itertools.product(bases, repeat=prefix_len):
                    self.npad_tokens.append(''.join(prefix) + 'N' * (self.k - prefix_len))

        # tail="char": single-base tokens for the trailing < k bases (NatureLM
        # IUPAC alphabet), so N / ambiguity codes fall back to a token, not <oov>.
        self.base_tokens = []
        self.dna_base_token = {}         # base -> token, e.g. "A" -> "<b_A>"
        self.dna_base_token_to_base = {}  # token -> base
        if self.tail == "char":
            for b in self.SINGLE_BASE_ALPHABET:
                tok = f"<b_{b}>"
                self.base_tokens.append(tok)
                self.dna_base_token[b] = tok
                self.dna_base_token_to_base[tok] = b

        # DNA tokens start after base vocabulary
        self.dna_start_id = self._base_vocab_size

        # All DNA tokens get new IDs (no reuse of base vocab IDs, even for
        # overlapping tokens like CCCCCC — they have different semantics in
        # DNA context vs BPE context, per Qiuyi's recommendation). Tail-variant
        # tokens are listed AFTER the 4^k k-mers so the k-mer ids are unchanged.
        base_dna_tokens = self.dna_special_tokens + self.kmers + self.npad_tokens + self.base_tokens

        # Calculate padding for 128 alignment
        total_vocab_unpadded = self._base_vocab_size + len(base_dna_tokens)
        target_vocab_size = ((total_vocab_unpadded + 127) // 128) * 128
        num_padding_tokens = target_vocab_size - total_vocab_unpadded

        # Add unused padding tokens
        self.padding_tokens = [f"<unused_{i}>" for i in range(num_padding_tokens)]

        # Create DNA token mappings — all get sequential new IDs
        self.dna_token_to_id = {}
        self.dna_id_to_token = {}

        current_id = self.dna_start_id
        for token in base_dna_tokens:
            self.dna_token_to_id[token] = current_id
            self.dna_id_to_token[current_id] = token
            current_id += 1

        # Add padding tokens
        for token in self.padding_tokens:
            self.dna_token_to_id[token] = current_id
            self.dna_id_to_token[current_id] = token
            current_id += 1

        self.dna_vocab_size = len(base_dna_tokens) + len(self.padding_tokens)

        # Set DNA special token IDs
        self.dna_begin_token_id = self.dna_token_to_id["<dna>"]
        self.dna_end_token_id = self.dna_token_to_id["</dna>"]
        self.oov_token_id = self.dna_token_to_id["<oov>"]

    def _build_combined_vocab(self):
        """Build combined vocabulary (base + DNA)."""
        self._vocab = self._base_vocab.copy()

        for token, token_id in self.dna_token_to_id.items():
            if token not in self._vocab:
                self._vocab[token] = token_id

        self._id_to_token = {v: k for k, v in self._vocab.items()}
        for token_id, token in self.dna_id_to_token.items():
            if token_id not in self._id_to_token:
                self._id_to_token[token_id] = token

    @property
    def vocab_size(self) -> int:
        return max(self._vocab.values()) + 1

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab.copy()

    def __len__(self):
        # Override default (len(get_vocab())) because get_vocab() deduplicates
        # CCCCCC which exists as both BPE (ID 91443) and DNA 6-mer (ID 154402).
        return self.vocab_size

    @classmethod
    def _auto_tag_text_segment(cls, text: str) -> str:
        """Tag conservative DNA candidates in a text-only segment.

        Each maximal IUPAC nucleotide run is considered independently. Pure
        A/C/G/T runs need at least six bases. Runs containing IUPAC ambiguity
        codes need at least twelve bases and may contain at most 20% ambiguity.
        Rejected ambiguous runs stay wholly on the BPE path rather than having
        any A/C/G/T substring tagged.
        """
        parts = []
        previous_end = 0

        for match in cls.IUPAC_RUN_RE.finditer(text):
            candidate = match.group(0)
            normalized = candidate.upper()
            ambiguous_count = sum(base not in cls.STANDARD_DNA_ALPHABET for base in normalized)
            is_standard_dna = ambiguous_count == 0 and len(normalized) >= 6
            is_conservative_iupac = (
                ambiguous_count > 0
                and len(normalized) >= 12
                and ambiguous_count / len(normalized) <= 0.2
            )

            if not (is_standard_dna or is_conservative_iupac):
                continue

            parts.append(text[previous_end:match.start()])
            parts.append(f"<dna>{normalized}</dna>")
            previous_end = match.end()

        if not parts:
            return text

        parts.append(text[previous_end:])
        return "".join(parts)

    @classmethod
    def _normalize_spaced_dna(cls, text: str) -> str:
        """Join uppercase DNA displays with at least six chunks and twelve bases.

        Chunks contain one to five A/C/G/T bases, separated only by spaces or
        tabs. Alphabetic boundaries prevent joining fragments of words.
        """
        def normalize(match):
            sequence = "".join(match.group(0).split())
            return sequence if len(sequence) >= 12 else match.group(0)

        return cls.SPACED_DNA_RE.sub(normalize, text)

    def _add_auto_dna_tags(self, text: str) -> str:
        """Tag DNA outside existing <dna>...</dna> regions."""
        return "".join(
            segment if is_dna else self._auto_tag_text_segment(self._normalize_spaced_dna(segment))
            for segment, is_dna in self._split_by_dna_tags(text)
        )

    def _split_by_dna_tags(self, text: str) -> List[Tuple[str, bool]]:
        segments = []
        i = 0
        n = len(text)

        while i < n:
            start_pos = text.find('<dna>', i)
            end_pos = text.find('</dna>', i)

            if start_pos == -1 and end_pos == -1:
                remaining = text[i:]
                if remaining:
                    segments.append((remaining, False))
                break

            if start_pos == -1 and end_pos != -1:
                dna_region = text[i:end_pos + 6]
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6
                continue

            if start_pos != -1 and end_pos == -1:
                if i < start_pos:
                    normal_text = text[i:start_pos]
                    if normal_text:
                        segments.append((normal_text, False))
                dna_region = text[start_pos:]
                if dna_region:
                    segments.append((dna_region, True))
                break

            if start_pos < end_pos:
                if i < start_pos:
                    normal_text = text[i:start_pos]
                    if normal_text:
                        segments.append((normal_text, False))
                dna_region = text[start_pos:end_pos + 6]
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6
            else:
                dna_region = text[i:end_pos + 6]
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6

        return segments

    def _parse_dna_region(self, dna_region: str) -> Tuple[str, bool, bool]:
        if dna_region == '<dna>':
            return '', True, False
        elif dna_region == '</dna>':
            return '', False, True

        has_start = dna_region.startswith('<dna>')
        has_end = dna_region.endswith('</dna>')

        content = dna_region
        if has_start:
            content = content[5:]
        if has_end and content.endswith('</dna>'):
            content = content[:-6]

        return content.strip(), has_start, has_end

    def _process_dna_sequence(self, dna_seq: str) -> Dict:
        k = self.k
        dna_seq = dna_seq.upper()

        kmer_tokens = []
        valid_bases = set('ATCG')

        def is_valid_kmer(kmer):
            return len(kmer) == k and all(base in valid_bases for base in kmer)

        if self.tail == "char":
            # Greedy NatureLM tokenization (NatureLM, arXiv:2502.07527), ported
            # from microsoft/NatureLM-8x7B nlm_tokenizer.py::_tokenize_entity:
            # take a full k-mer when the next k chars form one, else emit a single
            # base token and advance by 1 (also re-aligns after any non-base char).
            i = 0
            n = len(dna_seq)
            while i < n:
                window = dna_seq[i:i + k]
                if is_valid_kmer(window):
                    kmer_tokens.append(window)
                    i += k
                else:
                    base = dna_seq[i]
                    kmer_tokens.append(self.dna_base_token.get(base, "<oov>"))
                    i += 1
            return {"kmer_tokens": kmer_tokens, "padding_length": 0, "valid_length": k}

        for i in range(0, len(dna_seq) - k + 1, k):
            kmer = dna_seq[i:i+k]
            if is_valid_kmer(kmer):
                kmer_tokens.append(kmer)
            else:
                kmer_tokens.append("<oov>")

        processed_length = len(kmer_tokens) * k
        remaining = dna_seq[processed_length:]
        padding_length = 0
        valid_length = k

        if remaining:
            padding_needed = k - len(remaining)
            # Right-pad up to a full k-mer: real bases occupy positions
            # [0, valid_length). The hybrid BP loss supervises positions
            # 0..valid_len-1 via pos_mask = (valid_len > pos), so padding must be
            # at the END. pad_char="A" reuses a real k-mer id ("CT"->"CTAAAA");
            # "N" maps to a dedicated tail token ("CT"->"CTNNNN").
            padded = remaining + self.pad_char * padding_needed

            if all(base in valid_bases for base in remaining):
                kmer_tokens.append(padded)
            else:
                kmer_tokens.append("<oov>")

            padding_length = padding_needed
            valid_length = len(remaining)

        return {
            "kmer_tokens": kmer_tokens,
            "padding_length": padding_length,
            "valid_length": valid_length,
        }

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.dna_token_to_id:
            return self.dna_token_to_id[token]
        return self._base_vocab.get(token, self._base_tokenizer.unk_token_id or 0)

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.dna_id_to_token:
            return self.dna_id_to_token[index]
        return self._id_to_token.get(index, "<oov>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self.decode(ids, skip_special_tokens=False)
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_token_mask: bool = False,
        auto_dna_tags: Optional[bool] = None,
        **kwargs
    ) -> Union[List[int], Tuple[List[int], List[int]]]:
        use_auto = self.auto_dna_tags if auto_dna_tags is None else auto_dna_tags
        if use_auto:
            text = self._add_auto_dna_tags(text)

        segments = self._split_by_dna_tags(text)

        token_ids = []
        token_mask = [] if return_token_mask else None

        for segment_content, is_dna in segments:
            if is_dna:
                dna_content, has_start, has_end = self._parse_dna_region(segment_content)

                if has_start:
                    token_ids.append(self.dna_begin_token_id)
                    if return_token_mask:
                        token_mask.append(0)

                if dna_content:
                    result = self._process_dna_sequence(dna_content)

                    for idx, kmer in enumerate(result["kmer_tokens"]):
                        token_id = self.dna_token_to_id.get(kmer, self.oov_token_id)
                        token_ids.append(token_id)

                        if return_token_mask:
                            if kmer == "<oov>":
                                token_mask.append(0)
                            elif kmer in self.dna_base_token_to_base:
                                token_mask.append(1)  # char-fallback single base
                            elif idx == len(result["kmer_tokens"]) - 1 and result["padding_length"] > 0:
                                token_mask.append(result["valid_length"])
                            else:
                                token_mask.append(self.k)

                if has_end:
                    token_ids.append(self.dna_end_token_id)
                    if return_token_mask:
                        token_mask.append(0)
            else:
                base_ids = self._base_tokenizer.encode(
                    segment_content,
                    add_special_tokens=False
                )
                token_ids.extend(base_ids)
                if return_token_mask:
                    token_mask.extend([-1] * len(base_ids))

        # Do NOT append EOS when add_special_tokens=True. Qwen3 doesn't add
        # BOS/EOS either, and appending EOS here breaks lighteval's
        # tok_encode_pair: it relies on
        #   len(encode(ctx)) + len(encode(answer)) == len(encode(ctx + answer))
        # which the extra EOS violates by shifting the split by 1.

        if return_token_mask:
            return token_ids, token_mask
        return token_ids

    def _decode_dna_token(self, token_str: str) -> str:
        # Surface text for a DNA-vocab token. Baseline (pad_a) returns it
        # unchanged; char maps "<b_A>"->"A"; pad_n strips the N-pad tail fill
        # ("CTNNNN"->"CT"). Real k-mers contain no N, so rstrip is a no-op there.
        if token_str in self.dna_base_token_to_base:
            return self.dna_base_token_to_base[token_str]
        if self.tail == "pad_n" and len(token_str) == self.k and token_str[0] in "ATCG":
            return token_str.rstrip("N")
        return token_str

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> str:
        # A raw tensor/ndarray must become plain ints: dna_id_to_token is keyed by
        # python ints and a 0-dim tensor scalar hashes by identity, so DNA lookups
        # silently miss and the bases are dropped (e.g. decode(generate(...)[0])).
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            special_ids = {self.eos_token_id, self.pad_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        parts = []
        i = 0

        while i < len(token_ids):
            tid = token_ids[i]

            if tid == self.dna_begin_token_id:
                dna_tokens = []
                i += 1

                while i < len(token_ids) and token_ids[i] != self.dna_end_token_id:
                    if token_ids[i] in self.dna_id_to_token:
                        dna_tokens.append(self._decode_dna_token(self.dna_id_to_token[token_ids[i]]))
                    i += 1

                dna_seq = ''.join(dna_tokens)

                if skip_special_tokens:
                    parts.append(dna_seq)
                else:
                    parts.append(f"<dna>{dna_seq}")
                    if i < len(token_ids) and token_ids[i] == self.dna_end_token_id:
                        parts.append("</dna>")
                        i += 1

            elif tid in self.dna_id_to_token:
                # This branch handles k-mer tokens that appear without a <dna>
                # wrapper — the common generation case where <dna> was in the
                # prompt but only the generated portion is being decoded.
                # K-mer tokens are content, not special tokens, so always decode
                # them.  Only drop true DNA special tokens (<dna>, </dna>, <oov>)
                # when skip_special_tokens=True.
                is_dna_special = tid in (self.dna_begin_token_id, self.dna_end_token_id, self.oov_token_id)
                if not (skip_special_tokens and is_dna_special):
                    parts.append(self._decode_dna_token(self.dna_id_to_token[tid]))
                i += 1

            else:
                text_ids = []
                while i < len(token_ids):
                    curr_id = token_ids[i]
                    if curr_id in self.dna_id_to_token or curr_id == self.dna_begin_token_id:
                        break
                    text_ids.append(curr_id)
                    i += 1

                if text_ids:
                    decoded = self._base_tokenizer.decode(text_ids, skip_special_tokens=skip_special_tokens)
                    parts.append(decoded)

        return ''.join(parts)

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs
    ) -> List[str]:
        return [
            self.decode(
                seq.tolist() if hasattr(seq, 'tolist') else list(seq),
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
            for seq in sequences
        ]

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_mask: bool = False,
        auto_dna_tags: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        all_ids = []
        all_masks = [] if return_token_mask else None

        for t in texts:
            if return_token_mask:
                ids, mask = self.encode(t, add_special_tokens=add_special_tokens, return_token_mask=True, auto_dna_tags=auto_dna_tags)
                all_ids.append(ids)
                all_masks.append(mask)
            else:
                ids = self.encode(t, add_special_tokens=add_special_tokens, return_token_mask=False, auto_dna_tags=auto_dna_tags)
                all_ids.append(ids)

        if padding:
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)

            padded_ids = []
            attention_masks = []
            padded_token_masks = [] if return_token_mask else None

            for idx, ids in enumerate(all_ids):
                pad_len = max_len - len(ids)

                if pad_len > 0:
                    ids = ids + [self.pad_token_id] * pad_len
                    attn = [1] * (max_len - pad_len) + [0] * pad_len
                    if return_token_mask:
                        mask = all_masks[idx] + [-2] * pad_len
                else:
                    ids = ids[:max_len]
                    attn = [1] * max_len
                    if return_token_mask:
                        mask = all_masks[idx][:max_len]

                padded_ids.append(ids)
                attention_masks.append(attn)
                if return_token_mask:
                    padded_token_masks.append(mask)

            all_ids = padded_ids
            all_masks = padded_token_masks
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]

        result = {
            "input_ids": all_ids if is_batch else all_ids[0],
            "attention_mask": attention_masks if is_batch else attention_masks[0],
        }

        if return_token_mask:
            result["token_mask"] = all_masks if is_batch else all_masks[0]

        if return_tensors == "pt":
            import torch
            if is_batch:
                result["input_ids"] = torch.tensor(result["input_ids"])
                result["attention_mask"] = torch.tensor(result["attention_mask"])
                if return_token_mask:
                    result["token_mask"] = torch.tensor(result["token_mask"])
            else:
                result["input_ids"] = torch.tensor([result["input_ids"]])
                result["attention_mask"] = torch.tensor([result["attention_mask"]])
                if return_token_mask:
                    result["token_mask"] = torch.tensor([result["token_mask"]])

        return BatchEncoding(result, tensor_type=return_tensors)

    # Markers delimiting an assistant turn, used to build the assistant-token mask.
    # Qwen chat format; override on the instance for other bases (e.g. Gemma uses
    # "<start_of_turn>model\n" / "<end_of_turn>").
    ASSISTANT_HEADER = "<|im_start|>assistant\n"
    TURN_END = "<|im_end|>\n"

    def apply_chat_template(self, conversation, *args, return_assistant_tokens_mask: bool = False, **kwargs):
        """Completion-only assistant mask WITHOUT a fast tokenizer.

        HF derives the assistant mask from ``{% generation %}`` char-spans via
        ``BatchEncoding.char_to_token()``, which only fast (Rust) tokenizers implement.
        HybridDNATokenizer is a slow Python tokenizer (the DNA k-mer logic can't be a
        ``tokenizer.json``), so that path raises
        ``char_to_token() is not available when using Python based tokenizers``.

        Instead we tokenize normally and mark the tokens of every assistant turn -- from
        just after ``ASSISTANT_HEADER`` up to and including the next ``TURN_END``. This
        makes ``return_assistant_tokens_mask=True`` (and thus TRL ``assistant_only_loss``)
        work unchanged, and needs no ``{% generation %}`` markers in the template.
        """
        auto_dna_tags = kwargs.pop("auto_dna_tags", None)
        if auto_dna_tags is not None:
            tokenizer_kwargs = dict(kwargs.pop("tokenizer_kwargs", None) or {})
            tokenizer_kwargs["auto_dna_tags"] = auto_dna_tags
            kwargs["tokenizer_kwargs"] = tokenizer_kwargs

        if not return_assistant_tokens_mask:
            return super().apply_chat_template(conversation, *args, **kwargs)
        kwargs.pop("return_assistant_tokens_mask", None)
        want_tensors = kwargs.pop("return_tensors", None)
        kwargs["tokenize"] = True
        kwargs["return_dict"] = True
        enc = super().apply_chat_template(conversation, *args, return_tensors=None, **kwargs)
        seqs = enc["input_ids"]
        batched = len(seqs) > 0 and isinstance(seqs[0], (list, tuple))
        batch = seqs if batched else [seqs]
        hdr = self(self.ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
        end = self(self.TURN_END, add_special_tokens=False)["input_ids"]
        masks = []
        for ids in batch:
            ids = list(ids)
            m = [0] * len(ids)
            i = 0
            while i <= len(ids) - len(hdr):
                if ids[i:i + len(hdr)] == hdr:
                    j = i + len(hdr)
                    while j < len(ids) and ids[j:j + len(end)] != end:
                        m[j] = 1
                        j += 1
                    for k in range(len(end)):          # include the turn-end token(s)
                        if j + k < len(ids):
                            m[j + k] = 1
                    i = j + len(end)
                else:
                    i += 1
            masks.append(m)
        enc["assistant_masks"] = masks if batched else masks[0]
        if want_tensors is not None:
            enc = enc.convert_to_tensors(tensor_type=want_tensors)
        return enc

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer files with the wrapper's effective chat template.
        self._base_tokenizer.chat_template = self.chat_template
        self._base_tokenizer.eos_token = self.eos_token
        self._base_tokenizer.pad_token = self.pad_token
        self._base_tokenizer.save_pretrained(save_directory)

        # Save DNA config
        dna_config = {
            "k": self.k,
            "dna_start_id": self.dna_start_id,
            "dna_vocab_size": self.dna_vocab_size,
            "dna_special_tokens": self.dna_special_tokens,
            "auto_dna_tags": self.auto_dna_tags,
            "tail": self.tail,
        }

        dna_config_path = os.path.join(save_directory, "dna_config.json")
        with open(dna_config_path, "w", encoding="utf-8") as f:
            json.dump(dna_config, f, indent=2)

        # Update tokenizer_config.json with auto_map
        config_path = os.path.join(save_directory, "tokenizer_config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        config.update({
            "tokenizer_class": "HybridDNATokenizer",
            "auto_map": {
                "AutoTokenizer": ["tokenizer.HybridDNATokenizer", None]
            },
            "k": self.k,
            "auto_dna_tags": self.auto_dna_tags,
            "tail": self.tail,
        })

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Copy this tokenizer.py to save directory
        import shutil
        src_py = os.path.abspath(__file__)
        dst_py = os.path.join(save_directory, "tokenizer.py")
        if os.path.exists(src_py) and src_py != dst_py:
            shutil.copy2(src_py, dst_py)

        return (save_directory,)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        trust_remote_code=False,
        **kwargs,
    ):
        # Custom loader: PreTrainedTokenizer.from_pretrained is incompatible with this
        # tokenizer's __init__ (it re-passes bos_token etc.), so construct directly from
        # bio_config/dna_config. That bypasses standard metadata loading, so also read
        # chat_template from tokenizer_config.json (vLLM / apply_chat_template need it).
        from transformers.utils.hub import cached_file

        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        hub_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "local_files_only": local_files_only,
            "proxies": proxies,
            "revision": revision,
            "subfolder": subfolder,
            "token": token,
            "_commit_hash": commit_hash,
            "_raise_exceptions_for_missing_entries": False,
        }

        cfg = {}
        for _name in ("bio_config.json", "dna_config.json"):
            _p = cached_file(pretrained_model_name_or_path, _name, **hub_kwargs)
            if _p is not None:
                with open(_p, encoding="utf-8") as _f:
                    cfg = json.load(_f)
                break

        _init = {"base_tokenizer_path": pretrained_model_name_or_path}
        for _key in ("k", "tail", "auto_dna_tags"):
            if _key in cfg:
                _init[_key] = cfg[_key]

        tokenizer_config_path = cached_file(pretrained_model_name_or_path, "tokenizer_config.json", **hub_kwargs)
        if tokenizer_config_path is not None:
            with open(tokenizer_config_path, encoding="utf-8") as _f:
                tokenizer_config = json.load(_f)
            if tokenizer_config.get("chat_template"):
                _init["chat_template"] = tokenizer_config["chat_template"]
            for key in ("eos_token", "pad_token"):
                if tokenizer_config.get(key) is not None:
                    value = tokenizer_config[key]
                    _init[key] = value["content"] if isinstance(value, dict) else value

        chat_template_path = cached_file(pretrained_model_name_or_path, "chat_template.jinja", **hub_kwargs)
        if chat_template_path is not None:
            with open(chat_template_path, encoding="utf-8") as _f:
                _init["chat_template"] = _f.read()

        _init.update(kwargs)
        tokenizer = cls(**_init)
        # DNA ids start right after the base vocabulary, so a layout that differs from the saved one would misalign every
        # DNA embedding. Fail instead of tokenizing with shifted ids.
        for _key in ("dna_start_id", "dna_vocab_size"):
            if _key in cfg and cfg[_key] != getattr(tokenizer, _key):
                raise ValueError(
                    f"{pretrained_model_name_or_path} was saved with {_key}={cfg[_key]}, but the base tokenizer gives "
                    f"{getattr(tokenizer, _key)}, so its DNA token ids would not match the model."
                )
        return tokenizer
