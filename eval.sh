# Without Sequence Classification
python eval.py \
    --test_data data/nikl_sc_test.json \
    --spm_file tokenizers/kcbert_subword_bpe.model \
    --model_store_path checkpoints \
    --model_postfix kcbert_subword_bpe_transformer_small

python eval.py \
    --test_data data/nikl_sc_test.json \
    --spm_file tokenizers/kcbert_character.model \
    --model_store_path checkpoints \
    --model_postfix kcbert_character_transformer_small

# With Sequence Classification
python eval.py \
    --test_data data/nikl_sc_test.json \
    --spm_file tokenizers/kcbert_subword_bpe.model \
    --model_store_path checkpoints \
    --model_postfix kcbert_subword_bpe_transformer_small_seqcls

python eval.py \
    --test_data data/nikl_sc_test.json \
    --spm_file tokenizers/kcbert_character.model \
    --model_store_path checkpoints \
    --model_postfix kcbert_character_transformer_small_seqcls