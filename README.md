# KorCorr: Korean Spelling Auto-correction with Seqence-labeling based copy generation

# How to execute

## Data Preparation

This project uses "Korean Spelling Correction Corpus" Distributed by NIKL(National Institute of Korean Language).
According to the 'Terms of agreement', we are not allowed to publicly redistribute the corpus itself but provide only parsing tools.

1. Apply for the usage of the Korean Spelling Correction Corpus in [NIKL Modoo Corpus Homepage](https://corpus.korean.go.kr/).
2. Download Korean Spelling Correction Corpus, and copy two files(`EXSC...json`, `MXSC...json`) into `data/` folder.
3. Run `python preprocess_nikl_sc.py` to merge and maintain only necessary keys for these two files. This will generate `data/nikl_sc.json`.
4. Run `python split_dataset.py data/nikl_sc.json` to get train/dev/test split(8:1:1). This will generate randomly splitted `data/nikl_sc_train.json`, `data/nikl_sc_dev.json`, `data/nikl_sc_test.json`.

## Training

### Training tokenizer

To train the tokenizer, you may run:

```python spm_train.py [TRAIN_TXT_FILE] [KWARGS]```

List of keyword arguments are equal to the sentencepiece trainer script, and details are provied [here](https://github.com/google/sentencepiece#train-sentencepiece-model). Examples are provided within `./spm_train.sh`. Trained tokenizers for our study is given in the tokenizers folder.

For our project, we have used..
- [KcBERT training data](https://www.kaggle.com/datasets/junbumlee/kcbert-pretraining-corpus-korean-news-comments?resource=download)'s downsized version(`data/kcbert_pretrain_small.txt`). We used a reduced version due to RAM shortage when training sentencepiece.

### Training model

To train the model, you may run:

```
python train.py
        --train_data [TRAIN_DATA] --dev_data [DEV_DATA]
        --spm_file [SPM_FILE]
        --model_store_path [MODEL_STORE_PATH]
        --model_postfix [MODEL_POSTFIX]
        [KWARGS]
```

- `train_data`, `dev_data`: Train and Dev set. If you have followed the above *Data Preparation* instructions, these two files will be `data/nikl_sc_train.json` and `data/nikl_sc_dev.json`, respectively.
- `spm_file`: sentencepiece tokenizer you would want to use.
- `model_store_path`, `model_postfix`: model_postfix represents the model's name. Logs for training and evaluation scripts, and model checkpoints will be stored within `[MODEL_STORE_PATH]/[MODEL_POSTFIX]`.
- `kwargs`: Miscellaneous model size and training hyperparameters. Default model configuration represents "transformer-small" architecture(Vaswani, 2015).

## Evaluation

To evaluate the model, you may run:

```
python train.py
        --test_data [TEST_DATA]
        --spm_file [SPM_FILE]
        --model_store_path [MODEL_STORE_PATH]
        --model_postfix [MODEL_POSTFIX]
        [KWARGS]
```

- `test_data`: Test set. If you have followed the above *Data Preparation* instructions, these two files will be `data/nikl_sc_test.json`
> _IMPORTANT_: If you have used non-default model size values, you must again provide them as command line arguments when evaluating.

# Experiment Results

## Copy Generation

## Tokenization