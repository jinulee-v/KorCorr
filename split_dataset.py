import argparse
import json
import random

TRAIN=0.8
DEV=0.1
TEST=0.1

def main(args):
    with open(args.file, "r", encoding="UTF-8") as file:
        data = json.load(file)
        random.shuffle(data)
        train = data[:int(TRAIN*len(data))]
        dev = data[int(TRAIN*len(data)):int((TRAIN+DEV)*len(data))]
        test = data[int((TRAIN+DEV)*len(data)):]

        with open(args.file.replace(".json", "_train.json"), "w", encoding="UTF-8") as outfile:
            json.dump(train, outfile, indent=4, ensure_ascii=False)
        with open(args.file.replace(".json", "_dev.json"), "w", encoding="UTF-8") as outfile:
            json.dump(dev, outfile, indent=4, ensure_ascii=False)
        with open(args.file.replace(".json", "_test.json"), "w", encoding="UTF-8") as outfile:
            json.dump(test, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")

    args = parser.parse_args()
    main(args)