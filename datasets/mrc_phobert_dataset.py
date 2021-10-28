#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_dataset.py
import sys

import json
import torch
from utils.PhoBertBPETokenizer import PhoBertBPETokenizer
from torch.utils.data import Dataset


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """

    def __init__(self, json_path, tokenizer: PhoBertBPETokenizer, max_length: int = 258, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labels of NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id
        """
        data = self.all_data[item]
        tokenizer = self.tokenizer
        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        print(query)
        print(context)
        if self.is_chinese:
            context = "".join(context.split())
            end_positions = [x + 1 for x in end_positions]
        else:
            # add space offsets
            words = context.split()
            # Vị trí start, end ở mức ký tự
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        tokens, type_ids, offsets, words = tokenizer.encode(query, context, add_special_tokens=True)
        # tokens = query_context_tokens.ids  # Id của các từ [101,....,102]
        # type_ids = query_context_tokens.type_ids  # [0,0,0,0,,1,1,1,1,1,1] 0 là query, 1 là context
        # offsets = query_context_tokens.offsets  # [(idx bắt đầu từ 1, idx kết thúc từ 1),.....(n,n) )]

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue

            origin_offset2token_idx_start[token_start] = token_idx  # {start_index : index_word,......}
            origin_offset2token_idx_end[token_end] = token_idx  # {end_index : index_word,.....}

        new_start_positions = [origin_offset2token_idx_start[start] for start in
                               start_positions]  # Lấy ra index của các từ được đánh nhãn là start
        new_end_positions = [origin_offset2token_idx_end[end] for end in
                             end_positions]  # Lấy ra index của các từ được đánh nhãn là end

        # [0,0,0,0,...,1,1,1,0] biểu thị phần content
        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        if not self.is_chinese:
            for token_idx in range(len(tokens)):
                current_word_idx = words[token_idx]
                next_word_idx = words[token_idx + 1] if token_idx + 1 < len(tokens) else None
                prev_word_idx = words[token_idx - 1] if token_idx - 1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is [SEP]
        sep_token = tokenizer.photokenizer.convert_tokens_to_ids("</s>")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1
        return [
            torch.LongTensor(tokens), # Tokens của câu
            torch.LongTensor(type_ids), # Type_Ids của câu
            torch.LongTensor(start_labels), # Vị trí của token = 1 -> token đấy là bắt đầu của 1 nhãn
            torch.LongTensor(end_labels), # Vị trí của token=1 -> token đấy là kết thúc của 1 nhãn
            torch.LongTensor(start_label_mask), # Vị trí token=1 -> token đấy là bắt đầu của 1 từ
            torch.LongTensor(end_label_mask), # Vị trí token=1 -> token đấy là kết thúc của 1 từ
            match_labels, # Match start, end là 1 nhãn
            sample_idx,
            label_idx
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    from datasets.collate_functions import collate_to_max_length
    from torch.utils.data import DataLoader


    # en datasets
    bert_path = "E:\\Code\\Python\\mrc-for-flat-nested-ner\\data\\models\\phobert-base"
    # json_path = "/mnt/mrc/ace2004/mrc-ner.train"
    json_path = "E:\\Code\\Python\\mrc-for-flat-nested-ner\\data\\datasets\\ner-Covid19\\data\\new_type\\mrc-ner.train"
    is_chinese = False

    vocab_file = os.path.join(bert_path, "vocab.txt")
    merges_file = os.path.join(bert_path, "bpe.codes")
    tokenizer = PhoBertBPETokenizer(vocab_file=vocab_file, merges_file=merges_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese)

    dataloader = DataLoader(dataset, batch_size=1,
                            collate_fn=collate_to_max_length)

    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(
                *batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            print(f'start_lables: {start_labels.numpy().tolist()}')

            tmp_start_position = []
            for tmp_idx, tmp_label in enumerate(start_labels.numpy().tolist()):
                if tmp_label != 0:
                    tmp_start_position.append(tmp_idx)

            tmp_end_position = []
            for tmp_idx, tmp_label in enumerate(end_labels.numpy().tolist()):
                if tmp_label != 0:
                    tmp_end_position.append(tmp_idx)

            if not start_positions:
                continue
            print("=" * 20)
            for start, end in zip(start_positions, end_positions):
                print(str(sample_idx.item()),
                      str(label_idx.item()) + "\t" + tokenizer.photokenizer.decode(tokens[start: end + 1]))

            print("!!!" * 20)
            for start, end in zip(tmp_start_position, tmp_end_position):
                print(str(sample_idx.item()),
                      str(label_idx.item()) + "\t" + tokenizer.photokenizer.decode(tokens[start: end + 1]))


if __name__ == '__main__':
    run_dataset()
