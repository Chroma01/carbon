# Data mixture for the 300B midtraining run, sourced by ../slurm/midtrain_300b.slurm.
#
# Reads DNA_DIR, BIO_DIR, ENGLISH_DIR and TAIL from the environment and leaves the Megatron --data-path string in
# DATA_PATH. Weights are fractions of the whole run and must sum to 1.0; the launcher checks that.
#
#   55%  Carbon DNA    eukaryote 70 / mRNA 16 / mRNA-splice 4 / prokaryote 10
#   15%  bio text      p90 7 / rest 7 / synthetic 1
#   20%  English       FineWeb-Edu 8 / DCLM 8 / FinePDFs-Edu 4
#   10%  instruction   Nemotron IF-Chat v3 (chat + instruction following) and v2 (chat)
#
# Eukaryote weights follow the natural species distribution of the source corpus (vertebrate-other 42%,
# vertebrate-mammalian 27%, invertebrate 20%, plant 8%, fungi 3%, protozoa 0.5%).

# Spread each source's weight evenly across the shards in its folder.
# Skipping 0-byte shards matters: the tokenizer leaves empty *_tokens.bin files when it is given more tasks than
# data, and counting them splits a source's weight onto empty files, silently starving it. The weight sum still
# looks correct, so this does not show up unless you check for it here.
DATA_PATH=""
add(){ mapfile -t P < <(for f in "$1"/*_tokens.bin; do [ -s "$f" ] && [ -s "${f%.bin}.idx" ] && printf '%s\n' "${f%.bin}"; done)
  # Weight each shard by its size rather than splitting a source's weight evenly. The even split matched the
  # original run's sharding; re-tokenizing FineDNA gives different shard sizes, and an even split would then
  # over-read the small ones. By size, a source delivers its share whatever --n_tasks you used.
  if [ ${#P[@]} -eq 0 ]; then echo "ERROR: no usable shards in $1 (need non-empty *_tokens.bin with a matching .idx)" >&2; exit 1; fi
  W=$("${PYBIN:-python3}" -c "
import os, sys
w, paths = float(sys.argv[1]), sys.argv[2:]
sizes = [os.path.getsize(p + '.bin') for p in paths]
print(' '.join(f'{w * s / sum(sizes)} {p}' for s, p in zip(sizes, paths)))" "$2" "${P[@]}") \
    || { echo "ERROR: weight calc failed for $1" >&2; exit 1; }
  DATA_PATH="$DATA_PATH $W"; }

# 55% DNA, split 70 eukaryote / 16 mRNA / 4 mRNA-splice / 10 prokaryote.
# Eukaryote weights follow the natural species distribution of the source corpus.
add $DNA_DIR/eukaryote/gener_metadata_fungi_qwen3h_$TAIL                $("${PYBIN:-python3}" -c "print(0.55*0.70*0.029)")
add $DNA_DIR/eukaryote/gener_metadata_protozoa_qwen3h_$TAIL             $("${PYBIN:-python3}" -c "print(0.55*0.70*0.005)")
add $DNA_DIR/eukaryote/gener_metadata_plant_qwen3h_$TAIL                $("${PYBIN:-python3}" -c "print(0.55*0.70*0.080)")
add $DNA_DIR/eukaryote/gener_metadata_invertebrate_qwen3h_$TAIL         $("${PYBIN:-python3}" -c "print(0.55*0.70*0.197)")
add $DNA_DIR/eukaryote/gener_metadata_vertebrate_mammalian_qwen3h_$TAIL $("${PYBIN:-python3}" -c "print(0.55*0.70*0.268)")
add $DNA_DIR/eukaryote/gener_metadata_vertebrate_other_qwen3h_$TAIL     $("${PYBIN:-python3}" -c "print(0.55*0.70*0.421)")
add $DNA_DIR/mrna_pure_qwen3h_$TAIL                                     $("${PYBIN:-python3}" -c "print(0.55*0.16)")
add $DNA_DIR/mrna_splice_promoter_qwen3h_$TAIL                          $("${PYBIN:-python3}" -c "print(0.55*0.04)")
add $DNA_DIR/prokaryote_qwen3h_$TAIL                                    $("${PYBIN:-python3}" -c "print(0.55*0.10)")

# 15% bio text: 7% p90 (long-DNA-bearing web and literature), 7% the rest of the bio corpus, 1% synthetic.
add $BIO_DIR/p90_train/biorxiv_qwen3h_$TAIL                             $("${PYBIN:-python3}" -c "print(0.07*0.0336)")
add $BIO_DIR/p90_train/biostars_qwen3h_$TAIL                            $("${PYBIN:-python3}" -c "print(0.07*0.0012)")
add $BIO_DIR/p90_train/finepdfs_qwen3h_$TAIL                            $("${PYBIN:-python3}" -c "print(0.07*0.6545)")
add $BIO_DIR/p90_train/fineweb_qwen3h_$TAIL                             $("${PYBIN:-python3}" -c "print(0.07*0.0265)")
add $BIO_DIR/p90_train/pubmed_qwen3h_$TAIL                              $("${PYBIN:-python3}" -c "print(0.07*0.2842)")
add $BIO_DIR/rest_train/bio_rest_qwen3h_$TAIL                           0.07
add $BIO_DIR/gemini_fim_train/gemini_fim_qwen3h_$TAIL                   0.005
add $BIO_DIR/gemini_nonfim_train/gemini_nonfim_qwen3h_$TAIL             0.005

# 20% English + 10% instruction SFT (Nemotron ifchat-v3, reasoning traces preserved).
add $ENGLISH_DIR/fwedu_100B_megatron_qwen3h                             0.08
add $ENGLISH_DIR/dclm_100B_megatron_qwen3h                              0.08
add $ENGLISH_DIR/finepdfs_edu_100B_megatron_qwen3h                      0.04
add $ENGLISH_DIR/sft_nemotron_ifchat_v3_qwen3h_npad                     0.10
