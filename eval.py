from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import sentencepiece as spm

import bleu

from model.korcorr_model import KorCorrModel
from model.correction_dataset import CorrectionDataset


def main(args):
    torch.manual_seed(args.torch_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make checkpoint/log directory
    model_store_path = os.path.join(args.model_store_path, args.model_postfix)
    try:
        os.mkdir(model_store_path)
    except FileExistsError:
        if args.secure:
            prompt = input("WARNING: overwriting directory " + model_store_path + ". Continnue? (y/n)")
            if prompt != "y":
                exit()

    # Init logger
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(model_store_path, "eval.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(stdout_handler)
    # logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.spm_file)

    if args.eval_from_file is None:        
        # Load model
        model_load_path = model_store_path
        assert os.path.isdir(model_load_path)
        last_checkpoint = sorted([f for f in os.listdir(model_load_path) if f.endswith(".pt")], reverse=True)[0]
        model_load_path = os.path.join(model_load_path, last_checkpoint)

        model = KorCorrModel(
            vocab_size=tokenizer.piece_size(),
            label_size=3,
            hidden_dim=args.hidden_dim,
            encoder_layers=args.encoder_layers,
            encoder_heads=args.encoder_heads,
            encoder_pf_dim=args.encoder_pf_dim,
            encoder_dropout=args.encoder_dropout, 
            decoder_layers=args.decoder_layers,
            decoder_heads=args.decoder_heads,
            decoder_pf_dim=args.decoder_pf_dim,
            decoder_dropout=args.decoder_dropout,
            pad_idx=tokenizer.pad_id(),
            bos_idx=tokenizer.bos_id(),
            eos_idx=tokenizer.eos_id(),
            max_length=args.max_length,
            use_seqcls_decoding=args.use_seqcls_decoding,
            device=device
        ).to(device)
        model.load_state_dict(torch.load(model_load_path))
        model.device = device
        model = model.to(device)
        model.eval()
    else:
        with open(args.eval_from_file, "r", encoding="UTF-8") as file:
            eval_data = json.load(file)

    # Load data
    logger.info("Generating dataset...")
    with open(args.test_data, "r", encoding='UTF-8') as file:
        test_data = json.load(file)
    test_dataset = CorrectionDataset(test_data, tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate)
    logger.info("Done")

    # Define criteria and optimizer
    gen_criteria = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    def label_criteria(input, target, mask):
        loss = F.cross_entropy(input, target, reduction="none")
        loss *= mask
        return torch.sum(loss) / torch.sum(mask) # Mean averaging

    # Eval phase (on dev set)
    total = len(test_dataset)
    dev_loss = 0

    inputs = []
    outputs = []
    goldens = []

    label_total = 0
    label_correct = 0

    for original_data in test_dataset.data:
        inputs.append(original_data["form_str"])
        goldens.append(unicodedata.normalize('NFKC', original_data["corrected_form_str"]))
    if args.eval_from_file is None:
        with torch.no_grad():
            for batch in tqdm(test_loader):
                before, after, labels = batch
                before = before.to(device)
                after = after.to(device)
                labels = labels.to(device)
                # forward + backward + optimize
                output_labels, output_generation = model(before, after[:, :-1])
                loss = gen_criteria(output_generation.transpose(1, 2), after[:, 1:])
                if args.use_label_loss:
                    mask = (before!=tokenizer.pad_id()) * (before!=tokenizer.eos_id()) * (before!=tokenizer.bos_id())
                    loss += label_criteria(output_labels.transpose(1, 2), labels, mask)
                    label_total += torch.sum(mask).item()
                    output_labels = torch.argmax(output_labels, dim=2)
                    label_correct += torch.sum((output_labels == labels) * mask).item()
                dev_loss += loss

                outputs.extend(tokenizer.Decode(model.generate(before).tolist()))
    else:
        assert len(inputs) <= len(eval_data)
        eval_it = iter(eval_data)
        for i in inputs:
            while True:
                x = next(eval_it)
                if x["form"] == i:
                    break
            outputs.append(unicodedata.normalize('NFKC', x["corrected_form"]))
        assert len(inputs) == len(outputs)
    
    # bleu score
    nonl_goldens = [s.replace("\n", " ") for s in goldens]
    nonl_outputs = [s.replace("\n", " ") for s in outputs]
    bleu_score = bleu.list_bleu([nonl_goldens], nonl_outputs)

    # For calculating token precision and label accuracy,
    # We need to toke nize results and retireve sequence labels!
    test_result_data = [{
        "form": i,
        "corrected_form": o
    } for i, o in zip(inputs, outputs)]
    test_result_dataset = CorrectionDataset(test_result_data, tokenizer, max_length=args.max_length, filter_result_length=False)
    assert len(test_result_dataset) == len(test_dataset)

    # word precision, recall, F1
    word_true_positive = 0
    word_false_positive = 0
    word_false_negative = 0
    for i in range(len(test_dataset)):
        guess, goal = test_result_dataset[i], test_dataset[i]
        assert guess["form_str"] == goal["form_str"]
        tp_count = 0
        for token in guess["corrected_form_str"].split():
            if token in goal["corrected_form_str"].split():
                tp_count += 1
        word_true_positive += tp_count
        word_false_positive += len(guess["corrected_form_str"].split()) - tp_count
        word_false_negative += len(goal["corrected_form_str"].split()) - tp_count
    word_precision = word_true_positive / (word_true_positive + word_false_positive)
    word_recall = word_true_positive / (word_true_positive + word_false_negative)
    word_f1 = 2/(1/word_precision + 1/word_recall)

    # token precision, recall, F1
    token_true_positive = 0
    token_false_positive = 0
    token_false_negative = 0
    for i in range(len(test_dataset)):
        guess, goal = test_result_dataset[i], test_dataset[i]
        assert guess["form"].equal(goal["form"])
        tp_count = 0
        for token in guess["corrected_form"].tolist():
            if token in goal["corrected_form"].tolist():
                tp_count += 1
        token_true_positive += tp_count
        token_false_positive += guess["corrected_form"].size(0) - tp_count
        token_false_negative += goal["corrected_form"].size(0) - tp_count
    token_precision = token_true_positive / (token_true_positive + token_false_positive)
    token_recall = token_true_positive / (token_true_positive + token_false_negative)
    token_f1 = 2/(1/token_precision + 1/token_recall)

    # label accuracy
    total_labels = 0
    accurate_labels = 0
    for i in range(len(test_dataset)):
        guess, goal = test_result_dataset[i], test_dataset[i]
        assert guess["align_labels"].size(0) == goal["align_labels"].size(0)
        total_labels += guess["align_labels"].size(0)
        accurate_labels += torch.sum(guess["align_labels"] == goal["align_labels"]).item()


    logger.info("=================================================")
    logger.info(f"test loss = {dev_loss/total}")
    logger.info(f"BLEU score = {bleu_score}")
    logger.info(f"Word-phrase P-R")
    logger.info(f"  precision = {word_precision * 100}")
    logger.info(f"  recall = {word_recall * 100}")
    logger.info(f"  F1 = {word_f1 * 100}")
    logger.info(f"Token P-R")
    logger.info(f"  precision = {token_precision * 100}")
    logger.info(f"  recall = {token_recall * 100}")
    logger.info(f"  F1 = {token_f1 * 100}")
    logger.info(f"Label accuracy(generation) = {accurate_labels / total_labels * 100}")
    if args.use_label_loss:
        logger.info(f"Label accuracy(classification) = {label_correct / label_total * 100}")
    logger.info("")
    logger.info("Test generation result")
    for i in range(10):
        logger.info(f"Example {i}")
        logger.info(f"    input: {inputs[i]}")
        logger.info(f"    output: {outputs[i]}")
        logger.info(f"    golden: {goldens[i]}")
    logger.info("")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--eval_from_file", default=None)

    # Tokenizer
    parser.add_argument("--spm_file", required=True)

    # Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--encoder_pf_dim", type=int, default=512)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--decoder_pf_dim", type=int, default=512)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--use_label_loss", action="store_true", default=False)
    parser.add_argument("--use_seqcls_decoding", action="store_true", default=False)

    # Training args
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=3000)

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=True)
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--maintain_best_chkpt_only", default=False)
    parser.add_argument("--secure", default=False)

    args = parser.parse_args()
    main(args)