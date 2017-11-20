import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
from data_utils import Vocabulary
from data_utils import load_data_and_labels_klp, load_data_and_labels_exo
from konlpy.tag import Mecab

NER_idx_dic = {'<unk>':0, 'LC':1, 'DT':2, 'OG':3, 'TI':4, 'PS':5}

class DocumentDataset (data.Dataset):
    """"""
    def __init__(self, vocab, char_vocab, pos_vocab, lex_dict, x_text, x_split, x_pos, labels):
        """
        :param vocab:
        """
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab
        self.lex_dict = lex_dict
        self.x_text = x_text
        self.x_split = x_split
        self.x_pos = x_pos
        self.labels = labels

    def __getitem__(self, index):
        """Returns 'one' data pair """
        x_text_item = self.x_text[index]
        x_split_item = self.x_split[index]
        x_pos_item = self.x_pos[index]
        label_item = self.labels[index]

        x_text_char_item = []

        for x_word in x_text_item:
            x_char_item = []

            for x_char in x_word:
                x_char_item.append(x_char)

            x_text_char_item.append(x_char_item)



        x_idx_item = prepare_sequence(x_text_item, self.vocab.word2idx)
        x_idx_char_item = prepare_char_sequence(x_text_char_item, self.char_vocab.word2idx)
        x_pos_item = prepare_sequence(x_pos_item, self.pos_vocab.word2idx)
        x_lex_item = prepare_lex_sequence(x_text_item, self.lex_dict)





        label = torch.LongTensor(label_item)
        # print("label")
        # print(label)
        # print(type(label))

        return x_text_item, x_split_item, x_idx_item, x_idx_char_item, x_pos_item, x_lex_item, label

    def __len__(self):
        return len(self.x_text)

def prepare_sequence(seq, word_to_idx):
    idxs = list()
    # idxs.append(word_to_idx['<start>'])
    for word in seq:
        if word not in word_to_idx:
            idxs.append(word_to_idx['<unk>'])
        else:
            idxs.append(word_to_idx[word])
        # print(word_to_idx[word])
    # idxs.append(word_to_idx['<eos>'])
    return idxs

def prepare_char_sequence(seq, char_to_idx):
    char_idxs = list()
    # idxs.append(word_to_idx['<start>'])
    for word in seq:
        idxs = list()
        for char in word:
            if char not in char_to_idx:
                idxs.append(char_to_idx['<unk>'])
            else:
                idxs.append(char_to_idx[char])
        char_idxs.append(idxs)
        # print(word_to_idx[word])
    # idxs.append(word_to_idx['<eos>'])
    return char_idxs

def prepare_lex_sequence(seq, lex_to_ner_list):
    lex_idxs = list()
    # idxs.append(word_to_idx['<start>'])
    for lexicon in seq:
        if lexicon not in lex_to_ner_list:
            lex_idxs.append([lex_to_ner_list['<unk>']])
        else:
            lex_idxs.append(lex_to_ner_list[lexicon])
        # print(word_to_idx[word])
    # idxs.append(word_to_idx['<eos>'])
    return lex_idxs

def collate_fn(data):
    """Creates mini-batch tensor"""
    data.sort(key=lambda x: len(x[0]), reverse=True)

    x_text_batch, x_split_batch, x_idx_batch, x_idx_char_batch, x_pos_batch, x_lex_batch, labels = zip(*data)

    lengths = [len(label) for label in labels]
    targets = torch.zeros(len(labels), max(lengths), 10).long()
    for i, label in enumerate(labels):
        end = lengths[i]
        targets[i, :end] = label[:end]




    max_word_len = int(np.amax([len(word_tokens) for word_tokens in x_idx_batch])) # ToDo: usually, np.mean can be applied

    batch_size = len(x_idx_batch)

    batch_words_len = []
    batch_words_len = [len(word_tokens) for word_tokens in x_idx_batch]
    batch_words_len = np.array(batch_words_len)



    # Padding procedure (word)
    padded_word_tokens_matrix = np.zeros((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_word_tokens_matrix.shape[0]):
        for j in range(padded_word_tokens_matrix.shape[1]):
            try:
                padded_word_tokens_matrix[i, j] = x_idx_batch[i][j]
            except IndexError:
                pass

    max_char_len = int(np.amax([len(char_tokens) for word_tokens in x_idx_char_batch for char_tokens in word_tokens]))
    if max_char_len < 5: # size of maximum filter of CNN
        max_char_len = 5

    # Padding procedure (char)
    padded_char_tokens_matrix = np.zeros((batch_size, max_word_len, max_char_len), dtype=np.int64)
    for i in range(padded_char_tokens_matrix.shape[0]):
        for j in range(padded_char_tokens_matrix.shape[1]):
            for k in range(padded_char_tokens_matrix.shape[1]):
                try:
                    padded_char_tokens_matrix[i, j, k] = x_idx_char_batch[i][j][k]
                except IndexError:
                    pass

    # Padding procedure (pos)
    padded_pos_tokens_matrix = np.zeros((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_pos_tokens_matrix.shape[0]):
        for j in range(padded_pos_tokens_matrix.shape[1]):
            try:
                padded_pos_tokens_matrix[i, j] = x_pos_batch[i][j]
            except IndexError:
                pass


    # Padding procedure (lex)
    padded_lex_tokens_matrix = np.zeros((batch_size, max_word_len, len(NER_idx_dic)))
    for i in range(padded_lex_tokens_matrix.shape[0]):
        for j in range(padded_lex_tokens_matrix.shape[1]):
            for k in range(padded_lex_tokens_matrix.shape[2]):
                try:
                    for x_lex in x_lex_batch[i][j]:
                        k = NER_idx_dic[x_lex]
                        padded_lex_tokens_matrix[i, j, k] = 1
                except IndexError:
                    pass

    padded_word_tokens_matrix = torch.from_numpy(padded_word_tokens_matrix)
    padded_char_tokens_matrix = torch.from_numpy(padded_char_tokens_matrix)
    padded_pos_tokens_matrix = torch.from_numpy(padded_pos_tokens_matrix)
    padded_lex_tokens_matrix = torch.from_numpy(padded_lex_tokens_matrix).float()


    return x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, targets, batch_words_len


def get_loader(data_file_dir, vocab, char_vocab, pos_vocab, lex_dict, batch_size, shuffle, num_workers, dataset='klp'):
    """"""
    if dataset == 'klp':
        x_list, x_pos_list, x_split_list, y_list = load_data_and_labels_klp(data_file_dir=data_file_dir)
        y_list = np.array(y_list)
    elif dataset == 'exo':
        x_list, x_pos_list, x_split_list, y_list = load_data_and_labels_exo(data_file_dir='./data_in/EXOBRAIN_NE_CORPUS_10000.txt')
        y_list = np.array(y_list)
    elif dataset == 'both':
        x_list, x_pos_list, x_split_list, y_list = load_data_and_labels_klp(data_file_dir=data_file_dir)
        x_list_2, x_pos_list_2, x_split_list_2, y_list_2 = load_data_and_labels_exo(data_file_dir='./data_in/EXOBRAIN_NE_CORPUS_10000.txt')


        x_list = x_list + x_list_2
        x_pos_list = x_pos_list + x_pos_list_2
        x_split_list = x_split_list + x_split_list_2
        y_list = y_list + y_list_2
        y_list = np.array(y_list)


    print("len(x_list):",len(x_list))
    print("len(y_list):",len(y_list))



    document = DocumentDataset(vocab=vocab,
                               char_vocab=char_vocab,
                               pos_vocab=pos_vocab,
                               lex_dict=lex_dict,
                               x_text=x_list,
                               x_split=x_split_list,
                               x_pos=x_pos_list,
                               labels=y_list)

    data_loader = torch.utils.data.DataLoader(dataset=document,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader