import argparse
import sentencepiece as spm

def main(args):
    templates= '--input={} \
    --pad_id={} \
    --bos_id={} \
    --eos_id={} \
    --unk_id={} \
    --model_prefix={} \
    --vocab_size={} \
    --character_coverage={} \
    --model_type={}'

    cmd = templates.format(args.train_input_file,
                    args.pad_id,
                    args.bos_id,
                    args.eos_id,
                    args.unk_id,
                    args.prefix,
                    args.vocab_size,
                    args.character_coverage,
                    args.model_type)

    spm.SentencePieceTrainer.Train(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input_file")
    parser.add_argument("prefix")

    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--bos_id", type=int, default=1)
    parser.add_argument("--eos_id", type=int, default=2)
    parser.add_argument("--unk_id", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", choices=["unigram", "bpe", "char", "word"], default="unigram")

    args = parser.parse_args()
    main(args)