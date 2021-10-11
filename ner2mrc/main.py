class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bos_decode(char_label_list):
    print('char_label_list: ', char_label_list)
    idx = 0
    length = len(char_label_list)
    tags = []
    term_meger = ''
    start = 0
    end = 0
    # lặp từng kí tự
    while idx < length:
        # Lấy term, label hiện tại
        term, label = char_label_list[idx]
        current_label = label[0]

        # Nếu từ hiện tại có nhãn = O
        if current_label == 'O':
            idx += 1
            continue

        if current_label == 'B':
            start = idx
            end = idx
            term_meger = term_meger + ' ' + term

        if current_label == "I":
            end = idx
            term_meger = term_meger + ' ' + term

        # Lấy nhãn tiếp theo
        if idx + 1 == length:
            tags.append(Tag(term_meger.strip(), label[2:], start, end))
            term_meger = ''
        else:
            _, next_label = char_label_list[idx + 1]
            if next_label == 'O':
                tags.append(Tag(term_meger.strip(), label[2:], start, end))
                term_meger = ''

        idx += 1
    return tags


if __name__ == '__main__':
    x = [('Đồng_thời', 'O'), (',', 'O'), ('bệnh_viện', 'O'), ('tiếp_tục', 'O'), ('thực_hiện', 'O'), ('các', 'O'), ('biện_pháp', 'O'), ('phòng_chống', 'O'), ('dịch_bệnh', 'O'), ('COVID', 'O'), ('-', 'O'), ('19', 'O'), ('theo', 'O'), ('hướng_dẫn', 'O'), ('của', 'O'), ('Bộ', 'B-ORGANIZATION'), ('Y_tế', 'I-ORGANIZATION'), ('.', 'O')]
    tags = bos_decode(x)
    print(tags)
