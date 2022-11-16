# Without Sequence Classification
python train.py \
    --train_data data/nikl_sc_train.json \
    --dev_data data/nikl_sc_dev.json \
    --spm_file tokenizers/kcbert_subword_bpe.model \
    --model_store_path checkpoints \
    --model_postfix kcbert_subword_bpe_transformer_small

python train.py \
    --train_data data/nikl_sc_train.json \
    --dev_data data/nikl_sc_dev.json \
    --spm_file tokenizers/kcbert_character.model \
    --model_store_path checkpoints \
    --model_postfix kcbert_character_transformer_small

# With Sequence Classification
python train.py \
    --train_data data/nikl_sc_train.json \
    --dev_data data/nikl_sc_dev.json \
    --spm_file tokenizers/kcbert_subword_bpe.model \
    --use_label_loss \
    --model_store_path checkpoints \
    --model_postfix kcbert_subword_bpe_transformer_small_seqcls

python train.py \
    --train_data data/nikl_sc_train.json \
    --dev_data data/nikl_sc_dev.json \
    --spm_file tokenizers/kcbert_character.model \
    --use_label_loss \
    --model_store_path checkpoints \
    --model_postfix kcbert_character_bpe_transformer_small_seqcls