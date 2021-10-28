from transformers import PhobertTokenizer
import re


class PhoBertBPETokenizer():
    def __init__(self, vocab_file, merges_file):
        self.ids = []
        self.type_ids = []
        self.offsets = []
        self.words = []
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.photokenizer = PhobertTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)

    def encode(self, query, context, add_special_tokens):
        # Lấy ids query + context
        self.ids = self.photokenizer.encode(query, context, add_special_tokens=add_special_tokens)
        self.tokens = [self.photokenizer._convert_id_to_token(index=id) for id in self.ids]
        # print('self.tokens: ', self.photokenizer)

        # Lấy Type_ids
        index_sep_tags = [index for index, element in enumerate(self.ids) if element == 2]
        self.type_ids = [0] * len(self.ids)
        index = index_sep_tags[1]
        for i in range(index, len(self.ids)):
            self.type_ids[i] = 1

        # Lấy Offsets
        query_offsets = self.get_offset(query)
        context_offsets = self.get_offset(context)
        self.offsets = query_offsets + context_offsets
        # print(len(query), ' ', len(context))

        # Lấy Words của từng token
        query_words = self.get_words(query)
        context_words = self.get_words(context)
        self.words = [-1] + query_words + [-1] + [-1] + context_words + [-1]

        return self.ids, self.type_ids, self.offsets, self.words

    def get_offset(self, string):
        # Lấy offset của từng token
        data = string
        words = re.findall(r"\S+\n?", string)
        tokens = []
        for i in range(len(words)):
            tokens.extend([t for t in self.photokenizer.bpe(words[i]).split(" ")])
        tokens.insert(0, '<s>')
        tokens.insert(-1, '<//s>')
        offsets = []
        idx_current = 0
        for token in tokens:
            # Nếu các token là các token đặc biệt:
            if token == '<s>' or token == '<//s>':
                offsets.append((0, 0))
                continue

            token = token.replace("@@", "").strip()
            idx_start = data.find(token, idx_current)
            if idx_start == -1:
                print(f'token not found {token}')
                offsets.append((0, 0))
                continue
            idx_end = idx_start + len(token)
            offsets.append((idx_start, idx_end))
            idx_current = idx_end
        return offsets

    def get_words(self, string):
        idx_words = []
        words = re.findall(r"\S+\n?", string)
        for i in range(len(words)):
            tokens = [t for t in self.photokenizer.bpe(words[i]).split(" ")]
            for j in range(len(tokens)):
                idx_words.append(i)
        return idx_words