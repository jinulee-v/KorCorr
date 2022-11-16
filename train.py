from argparse import ArgumentParser
import json
from tqdm import tqdm
import os, sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import sentencepiece as spm

from model.korcorr_model import KorCorrModel
from model.correction_dataset import CorrectionDataset

model_id = "facebook/bart-base"


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
    file_handler = logging.FileHandler(os.path.join(model_store_path, "train.log"))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Log basic info
    logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info("- %s: %r", arg, value)
    logger.info("")

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.LoadFromFile(args.spm_file)
    
    # Load model
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
        device=device
    ).to(device)
    # if args.from_checkpoint is not None:
    #     assert os.path.isdir(args.model_store_path)
    #     model_load_path = os.path.join(args.model_store_path, args.from_checkpoint)
    #     assert os.path.isdir(model_load_path)
    #     last_checkpoint = sorted([f for f in os.listdir(model_load_path) if f.endswith(".pt")], reverse=True)[0]
    #     model_load_path = os.path.join(model_load_path, last_checkpoint)
    #     model.load_state_dict(torch.load(model_load_path))
    #     model.device = device
    #     model = model.to(device)

    # Load data
    logger.info("Generating dataset...")
    with open(args.train_data, "r", encoding='UTF-8') as file:
        train_data = json.load(file)
    with open(args.dev_data, "r", encoding='UTF-8') as file:
        dev_data = json.load(file)
    train_dataset = CorrectionDataset(train_data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate)
    dev_dataset = CorrectionDataset(dev_data, tokenizer, max_length=args.max_length)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate)
    logger.info("Done")

    # Define criteria and optimizer
    gen_criteria = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    def label_criteria(input, target, mask):
        loss = F.cross_entropy(input, target, reduction="none")
        loss *= mask
        return torch.sum(loss) / torch.sum(mask) # Mean averaging
    optimizer = Adam(model.parameters(), lr=args.lr)

    min_loss = 1e+10
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        logger.info(f"< epoch {epoch} >")
        # Train phase
        model.train()
        epoch_size = len(train_loader)
        for i, batch in enumerate(tqdm(train_loader)):
            before, after, labels = batch
            before = before.to(device)
            after = after.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_labels, output_generation = model(before, after[:, :-1])
            loss = gen_criteria(output_generation.transpose(1, 2), after[:, 1:])

            if args.use_label_loss:
                mask = (before!=tokenizer.pad_id()) * (before!=tokenizer.eos_id()) * (before!=tokenizer.bos_id())
                loss += label_criteria(output_labels.transpose(1, 2), labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if i % args.log_interval == args.log_interval-1 or i == epoch_size-1:
                # Eval phase (on dev set)
                model.eval()
                total = len(dev_data)
                dev_loss = 0
                first_batch=True

                with torch.no_grad():
                    for batch in tqdm(dev_loader):
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
                        dev_loss += loss

                        if first_batch:
                            first_batch=False
                            test_input = tokenizer.DecodeIds(before[0].tolist())
                            test_output = model.generate(before)[0]
                            test_output = tokenizer.DecodeIds(test_output.tolist())

                logger.info("=================================================")
                logger.info(f"epoch {epoch}, step {i}")
                logger.info(f"dev loss = {dev_loss/total}")
                logger.info("")
                logger.info("Test generation result")
                logger.info(f"input: {test_input}")
                logger.info(f"output: {test_output}")
                logger.info("")
                if dev_loss/total < min_loss:
                    logger.info(f"Updating min_loss = {min_loss} -> {dev_loss/total}")
                    min_loss = dev_loss / total
                    if args.maintain_best_chkpt_only:
                        os.remove(os.path.join(model_store_path, name))
                    logger.info("Save model checkpoint because reduced loss...")
                    name = f"KorCorr_{args.model_postfix}_epoch_{epoch}_step_{i+1}.pt"
                    torch.save(model.state_dict(), os.path.join(model_store_path, name))
                logger.info("=================================================")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Dataset
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--dev_data", required=True)

    #
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

    # Training args
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=3000)

    # Checkpoint configs
    parser.add_argument("--model_store_path", required=True)
    parser.add_argument("--model_postfix", required=True)
    parser.add_argument("--maintain_best_chkpt_only", default=False)
    parser.add_argument("--secure", default=False)

    args = parser.parse_args()
    main(args)