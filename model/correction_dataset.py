from itertools import product
from collections import deque
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor

kroneker = lambda a, b: int(a==b)

def needleman_wunsch(x: list, y: list):
    """
    based on https://johnlekberg.com/blog/2020-10-25-seq-align.html
    """
    N, M = len(x), len(y)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + kroneker(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    # In format of..
    # List(tuple)
    return list(alignment)


class CorrectionDataset():

    def __init__(self, raw_data, tokenizer: SentencePieceProcessor, filter_result_length:bool=True, max_length:int=512):
        """
        raw_data: List[Dict]. Same format as json training file
        """
        self.tokenizer = tokenizer

        # Reorganize dataset
        before_str = [pair["form"] for pair in raw_data]
        after_str = [pair["corrected_form"] for pair in raw_data]
        assert len(before_str) == len(after_str)
        before = [[tokenizer.bos_id()] + tok + [tokenizer.eos_id()] for tok in tokenizer.EncodeAsIds(before_str)]
        after = [[tokenizer.bos_id()] + tok + [tokenizer.eos_id()] for tok in tokenizer.EncodeAsIds(after_str)]

        # Align tokens
        align_labels = []
        for b, a in zip(tqdm(before), after):
            raw_alignment = needleman_wunsch(b, a)
            """
            0: OK                     ()
            1: INSERT NEXT            (If before's next token is OK)
            2: DELETE/REPLACE THIS    (Should delete consequent tokens with this label)
            
            before: 1 2   5 6 7   9 10
            label:  0 2   0 2 1   0  0
            after:  1 3 4 5   7 8 9 10
            """
            align_label = []
            for i, (btok, atok) in enumerate(raw_alignment):
                if btok is not None:
                    # Get token vocab idx
                    btok = b[btok]
                    if atok is not None:
                        atok = a[atok]
                    # Generate label
                    label = 0 if kroneker(btok, atok) else 2
                    if label == 0 and i < len(raw_alignment)-1 and raw_alignment[i+1][0] is None:
                        label = 1
                    align_label.append(label)
            assert len(align_label) == len(b)
            align_labels.append(align_label)
        assert len(align_labels) == len(before)

        if filter_result_length:
            self.data = [{
                "form_str": bstr,
                "form": torch.tensor(b),
                "corrected_form_str": astr,
                "corrected_form": torch.tensor(a),
                "align_labels": torch.tensor(l)
            } for b, a, l, bstr, astr in zip(before, after, align_labels, before_str, after_str) if len(b) < max_length and len(a) < max_length]
        else:
            self.data = [{
                "form_str": bstr,
                "form": torch.tensor(b),
                "corrected_form_str": astr,
                "corrected_form": torch.tensor(a),
                "align_labels": torch.tensor(l)
            } for b, a, l, bstr, astr in zip(before, after, align_labels, before_str, after_str) if len(b) < max_length]
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        return self.data[index]
    
    def collate(self, batch):
        before = [item["form"] for item in batch]
        after = [item["corrected_form"] for item in batch]
        labels = [item["align_labels"] for item in batch]
        before = pad_sequence(before, batch_first=True, padding_value=self.tokenizer.pad_id())
        after = pad_sequence(after, batch_first=True, padding_value=self.tokenizer.pad_id())
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        assert before.size() == labels.size()
        return before, after, labels
