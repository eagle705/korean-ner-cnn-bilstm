import numpy as np
import os

# from konlpy.tag import Kkma
# from konlpy.tag import Twitter
from konlpy.tag import Mecab

from collections import Counter
import pickle
import codecs
import argparse


import re


mecab = Mecab()

class Vocabulary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        self.idx += 1

    def __len__(self):
        return len(self.word2idx)

def build_vocab(text_list, threshold):
    """Build a simple vocab"""
    counter = Counter()
    # tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    for i, text in enumerate(text_list):
        print(text)
        # text = text.strip()
        # text = text.lower()

        # ToDo: English
        # tokens_en = nltk.word_tokenize(text)
        # tokens_en = mecab.pos(text)
        counter.update(text)





        if i % 1000 == 0:
            print("[%d/%d] Tokenized input text." %(i, len(text_list)))

    # words = [word for word, cnt in counter.items() if cnt >= threshold]
    words = [word for word, cnt in counter.items()]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<eos>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)




    print("Voca_size: ",len(vocab))
    print(vocab.idx2word)




    return vocab

def build_char_vocab(text_list, threshold):
    """Build a simple vocab"""
    counter = Counter()
    # tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    for i, text in enumerate(text_list):
        for word in text:
            for char in word:
                counter.update(char)


    # words = [word for word, cnt in counter.items() if cnt >= threshold]
    chars = [char for char, cnt in counter.items()]

    char_vocab = Vocabulary()
    char_vocab.add_word('<pad>')
    char_vocab.add_word('<unk>')

    for i, char in enumerate(chars):
        char_vocab.add_word(char)



    print("Char_Voca_size: ",len(char_vocab))
    print(char_vocab.idx2word)


    return char_vocab


def load_data_interactive(input_str):

    # Load data_in from files
    x_mor_list = list()
    x_pos_list = list()
    x_split_list = list()

    lines = [input_str]

    re_word = re.compile('<(.+?):[A-Z]{2}>')

    for line in lines:
        line = line.strip() #좌우 공백 제거


        raw_data = line
        split_raw_data = raw_data.split(' ')
        pos_data = mecab.pos(raw_data)

        x_split = []
        x_mor = []
        x_pos = []

        i = 0
        len_pos_word = 0
        len_split_word = 0
        for mor_pos in pos_data:
            if mor_pos[0] in split_raw_data[i]:
                len_pos_word += len(mor_pos[0])
                len_split_word = len(split_raw_data[i])

                # new_pos_data.append([i, pos_word[0], pos_word[1]])

                x_split.append(i)
                x_mor.append(mor_pos[0])
                x_pos.append(mor_pos[1])



                if len_pos_word == len_split_word:
                    i = i + 1
                    len_pos_word = 0
                    len_split_word = 0

        if len(x_mor) == 0: #mecab에러인지.. 가끔 하나가 빠짐 그거 제외
            continue

        x_mor_list.append(x_mor)
        x_pos_list.append(x_pos)
        x_split_list.append(x_split)

    return x_mor_list, x_pos_list, x_split_list


def load_data_and_labels_exo(data_file_dir):

    # Load data_in from files
    x_mor_list = list()
    x_pos_list = list()
    x_split_list = list()
    y_list = list()



    file_obj = codecs.open(data_file_dir, "r", "utf-8" )
    lines = file_obj.readlines()

    NER_label_list = [':PS',':DT',':LC',':OG',':TI']
    NER_dict = {'<PAD>': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                '<START>':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                '<STOP>':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'B_LC':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'B_DT': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'B_OG': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'B_TI': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'B_PS': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


    re_word = re.compile('<(.+?):[A-Z]{2}>')

    for line in lines:
        line = line.strip()

        raw_data = line.replace('<','').replace('>','').replace(':PS','').replace(':DT','').replace(':LC','').replace(':OG','').replace(':TI','')
        split_raw_data = raw_data.split(' ')
        pos_data = mecab.pos(raw_data)

        x_split = []
        x_mor = []
        x_pos = []

        i = 0
        len_pos_word = 0
        len_split_word = 0
        for mor_pos in pos_data:
            if mor_pos[0] in split_raw_data[i]:
                len_pos_word += len(mor_pos[0])
                len_split_word = len(split_raw_data[i])

                # new_pos_data.append([i, pos_word[0], pos_word[1]])

                x_split.append(i)
                x_mor.append(mor_pos[0])
                x_pos.append(mor_pos[1])



                if len_pos_word == len_split_word:
                    i = i + 1
                    len_pos_word = 0
                    len_split_word = 0

        if len(x_mor) == 0: #mecab에러인지... 가끔 하나가 빠짐 그거 제외
            continue

        x_mor_list.append(x_mor)
        x_pos_list.append(x_pos)
        x_split_list.append(x_split)



        # label data
        label_data = line
        label_split_data = label_data.split(' ')

        re_result = re_word.finditer(label_data)
        raw_re_word_list = []
        temp_re_word_list = []
        re_NER_list = []
        for re_result_item in re_result:
            re_NER_list.append(re_result_item.group()[-3:-1])
            raw_re_word_list.append(re_word.findall((re_result_item.group())))
            temp_re_word_list.append(re_word.findall((re_result_item.group()[1:])))
            for i, temp_re_word_item in enumerate(temp_re_word_list):
                if len(temp_re_word_item) != 0:
                    raw_re_word_list[i] = temp_re_word_item


        # re_NER_list = re_NER.findall(label_data)
        re_word_list = [[re_word[0].replace(' ', '')] for re_word in raw_re_word_list]
        # print("re_word_list:",re_word_list)

        y_data = ['O'] * len(x_mor)
        B_flag = 0
        data_len = 0
        B_I_data_len = 0



        for i in range(len(x_mor)):
            pos_i_split = x_split[i]
            word_mor = x_mor[i]
            pos = x_pos[i]

            if len(re_word_list) == 0:
                continue

            if word_mor in re_word_list[0][0]:


                # print("word_mor:", word_mor)
                # print("data_len:", data_len)
                # print("B_I_data_len:", B_I_data_len)

                if B_flag == 0 and re_word_list[0][0].startswith(word_mor):

                    data_len += len(word_mor)
                    B_I_data_len = len(re_word_list[0][0])

                    y_data[i] = 'B_'+re_NER_list[0]
                    B_flag = 1 # B_ token mark

                    if data_len == B_I_data_len:
                        re_word_list.pop(0)
                        re_NER_list.pop(0)

                        data_len = 0
                        B_I_data_len = 0
                        B_flag = 0 # B_ token mark init


                    elif i + 1 < len(x_mor):
                        if x_mor[i + 1] not in re_word_list[0][0]:  # 시작일줄 알았는데 서브스트링이고, 매칭도 안되고 다음글자가 속하지 않으면 다시 리셋
                            y_data[i] = 'O'
                            B_flag = 0
                            data_len = 0
                            B_I_data_len = 0
                            B_flag = 0  # B_ token mark init


                elif B_flag == 1:

                    data_len += len(word_mor)
                    B_I_data_len = len(re_word_list[0][0])

                    if data_len != B_I_data_len:
                        y_data[i] = 'I'
                    elif data_len == B_I_data_len:
                        y_data[i] = 'I'
                        re_word_list.pop(0)
                        re_NER_list.pop(0)
                        data_len = 0
                        B_I_data_len = 0
                        B_flag = 0

        # print("y_data: ", y_data)
        y_data_idx = []
        for y in y_data:
            y_data_idx.append(NER_dict[y])
        y_list.append(y_data_idx)


    #y_list = np.array(y_list)


    return x_mor_list, x_pos_list, x_split_list, y_list

def load_data_and_labels_klp(data_file_dir):

    # Load data_in from files
    x_mor_list = list()
    x_pos_list = list()
    x_split_list = list()
    y_list = list()


    file_obj = codecs.open(data_file_dir, "r", "utf-8" )
    lines = file_obj.readlines()

    NER_label_list = [':PS',':DT',':LC',':OG',':TI']
    NER_dict = {'<PAD>': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                '<START>':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                '<STOP>':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'B_LC':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'B_DT': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'B_OG': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'B_TI': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'B_PS': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    re_word = re.compile('<(.+?):[A-Z]{2}>')

    for line in lines:
        line = line.strip() #좌우 공백 제거

        if len(line) == 0:
            continue

        elif line[0] == ';': # raw data
            raw_data = line.replace('; ','')
            split_raw_data = raw_data.split(' ')
            pos_data = mecab.pos(raw_data)

            x_split = []
            x_mor = []
            x_pos = []

            i = 0
            len_pos_word = 0
            len_split_word = 0
            for mor_pos in pos_data:
                if mor_pos[0] in split_raw_data[i]:
                    len_pos_word += len(mor_pos[0])
                    len_split_word = len(split_raw_data[i])

                    # new_pos_data.append([i, pos_word[0], pos_word[1]])

                    x_split.append(i)
                    x_mor.append(mor_pos[0])
                    x_pos.append(mor_pos[1])



                    if len_pos_word == len_split_word:
                        i = i + 1
                        len_pos_word = 0
                        len_split_word = 0

            if len(x_mor) == 0:  # mecab에러인지... 가끔 하나가 빠짐 그거 제외
                continue

            x_mor_list.append(x_mor)
            x_pos_list.append(x_pos)
            x_split_list.append(x_split)

            # print("x_mor", x_mor)


        elif line[0] == '$': # label data
            label_data = line.replace('$','')
            # print("label_data: ",label_data)
            label_split_data = label_data.split(' ')

            re_result = re_word.finditer(label_data)
            raw_re_word_list = []
            temp_re_word_list = []
            re_NER_list = []
            for re_result_item in re_result:
                re_NER_list.append(re_result_item.group()[-3:-1])
                raw_re_word_list.append(re_word.findall((re_result_item.group())))
                temp_re_word_list.append(re_word.findall((re_result_item.group()[1:])))
                for i, temp_re_word_item in enumerate(temp_re_word_list):
                    if len(temp_re_word_item) != 0:
                        raw_re_word_list[i] = temp_re_word_item


            # re_NER_list = re_NER.findall(label_data)
            re_word_list = [[re_word[0].replace(' ', '')] for re_word in raw_re_word_list]
            # print("re_word_list:",re_word_list)

            y_data = ['O'] * len(x_mor)
            B_flag = 0
            data_len = 0
            B_I_data_len = 0


            for i in range(len(x_mor)):
                pos_i_split = x_split[i]
                word_mor = x_mor[i]
                pos = x_pos[i]

                if len(re_word_list) == 0:
                    continue

                if word_mor in re_word_list[0][0]:


                    # print("word_mor:", word_mor)
                    # print("data_len:", data_len)
                    # print("B_I_data_len:", B_I_data_len)

                    if B_flag == 0 and re_word_list[0][0].startswith(word_mor):

                        data_len += len(word_mor)
                        B_I_data_len = len(re_word_list[0][0])

                        y_data[i] = 'B_' + re_NER_list[0]
                        B_flag = 1  # B_ token mark

                        if data_len == B_I_data_len:
                            re_word_list.pop(0)
                            re_NER_list.pop(0)

                            data_len = 0
                            B_I_data_len = 0
                            B_flag = 0  # B_ token mark init

                        elif i+1 < len(x_mor):
                            if x_mor[i + 1] not in re_word_list[0][0]:  # 시작일줄 알았는데 서브스트링이고, 매칭도 안되고 다음글자가 속하지 않으면 다시 리셋
                                y_data[i] = 'O'
                                B_flag = 0
                                data_len = 0
                                B_I_data_len = 0
                                B_flag = 0  # B_ token mark init


                    elif B_flag == 1:

                        data_len += len(word_mor)
                        B_I_data_len = len(re_word_list[0][0])

                        if data_len != B_I_data_len:
                            y_data[i] = 'I'
                        elif data_len == B_I_data_len:
                            y_data[i] = 'I'
                            re_word_list.pop(0)
                            re_NER_list.pop(0)
                            data_len = 0
                            B_I_data_len = 0
                            B_flag = 0

            # print("y_data: ", y_data)
            y_data_idx = []
            for y in y_data:
                y_data_idx.append(NER_dict[y])
            y_list.append(y_data_idx)


    #y_list = np.array(y_list)


    return x_mor_list, x_pos_list, x_split_list, y_list



def load_lexicon_NER(data_file_dir):

    # Load data_in from files
    lexicon_list = list()
    NER_multi_list = list()


    file_obj = codecs.open(data_file_dir, "r", "utf-8" )
    lines = file_obj.readlines()


    for line in lines:
        line = line.strip() #좌우 공백 제거
        lexicon, ner_label = line.split('\t')
        lexicon_list.append(lexicon)
        ner_label_list = ner_label.split(',')
        NER_multi_list.append(ner_label_list)


    return lexicon_list, NER_multi_list


def plot_word_embeddng(wv_model_ko):


    embedding_weights = wv_model_ko.wv.syn0
    final_embeddings = embedding_weights
    labels = wv_model_ko.wv.index2word

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, rc

    print("font_list: ", font_manager.get_fontconfig_fonts())
    font_name = font_manager.FontProperties(fname='/Library/Fonts/NanumSquareBold.ttf').get_name()
    rc('font', family=font_name)

    def plot_with_labels(low_dim_embs, labels, filename='./data_out/tsne_' + str(args.word2vec_dim) + '.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [labels[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


def generate_word_embedding(x_list):

    from gensim.models import word2vec
    import multiprocessing
    import time

    print("multiprocessing.cpu_count(): ",multiprocessing.cpu_count())


    config = {
        'min_count': 0,  # 등장 횟수가 5 이하인 단어는 무시
        'size': args.word2vec_dim,  # 50차원짜리 벡터스페이스에 embedding
        'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용
        'batch_words': 1000,  # 사전을 구축할때 한번에 읽을 단어 수
        'iter': 8,  # 7,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수를 의미 #너무 오래 걸릴땐 좀 낮춰야
        'workers': multiprocessing.cpu_count() #윈도우에서 에러
    }


    docs_ko = x_list

    wv_model_ko = word2vec.Word2Vec(**config)
    count_t = time.time()

    wv_model_ko.build_vocab(docs_ko)
    print(wv_model_ko.corpus_count)
    wv_model_ko.train(docs_ko, total_examples=wv_model_ko.corpus_count, epochs=3)

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<eos>')
    vocab.add_word('<unk>')

    for index, word in enumerate(wv_model_ko.wv.index2word):
        vocab.add_word(word)

    word2vec_matrix = wv_model_ko.wv.syn0
    word2vec_matrix = np.concatenate((np.zeros((4, args.word2vec_dim)), word2vec_matrix), axis=0)
    wv_model_ko.wv.syn0 = word2vec_matrix

    print('Running Time : %.02f' % (time.time() - count_t))
    wv_model_ko.save('./data_in/word2vec/ko_word2vec_' + str(args.word2vec_dim) + '.model')

    # print(word2vec_matrix[0:5])
    print(word2vec_matrix.shape)
    print(len(vocab))

    # pprint(wv_model_en['man'])
    # pprint(wv_model_en.most_similar('man'))
    #plot_word_embeddng(wv_model_ko)

    return vocab, wv_model_ko


def main(args):


    train_data_path = args.data_file_dir_train
    # test_data_path = args.data_file_dir_test
    vocab_path = args.vocab_path
    threshold = args.threshold

    x_list, x_pos_list, x_split_list, y_list = load_data_and_labels_klp(train_data_path)
    x_list_2, x_pos_list_2, x_split_list_2, y_list_2 = load_data_and_labels_exo('./data_in/EXOBRAIN_NE_CORPUS_10000.txt')


    x_list = x_list + x_list_2
    x_pos_list = x_pos_list + x_pos_list_2
    x_split_list = x_split_list + x_split_list_2
    y_list = y_list + y_list_2
    y_list = np.array(y_list)

    # vocab = build_vocab(x_list, threshold=threshold)
    char_vocab = build_char_vocab(x_list, threshold=threshold)


    lexicon_list, NER_double_list = load_lexicon_NER('./data_in/gazette/korean_gazette')

    lex_dict = {'<unk>': '<unk>'}
    for i, lex in enumerate(lexicon_list):
        print(NER_double_list[i])
        lex_dict[lex] = NER_double_list[i]


    with open(args.lex_dict_path, 'wb') as f:
        pickle.dump(lex_dict, f)



    vocab, wv_model_ko = generate_word_embedding(x_list=x_list)


    counter = Counter()
    for i, pos in enumerate(x_pos_list):
        counter.update(pos)

    pos_words = [pos for pos, cnt in counter.items()]

    pos_vocab = Vocabulary()
    pos_vocab.add_word('<pad>')
    pos_vocab.add_word('<start>')
    pos_vocab.add_word('<eos>')
    pos_vocab.add_word('<unk>')


    for i, word in enumerate(pos_words):
        pos_vocab.add_word(word)

    print(vocab.idx2word)
    print(char_vocab.idx2word)
    print(pos_vocab.idx2word)
    print("len(vocab.idx2word):",vocab.idx2word)
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(args.pos_vocab_path, 'wb') as f:
        pickle.dump(pos_vocab, f)
    with open(args.char_vocab_path, 'wb') as f:
        pickle.dump(char_vocab, f)

    print("Total vocabulary size: %d" %len(vocab))
    print("Saved vocab to '%s'" %vocab_path)

    # # print(vocab.word2idx)
    # # {'<pad>': 0, '<unk>': 1, 'bromwell': 2, 'high': 3, 'is': 4, 'a': 5, 'comedy': 6, '.': 7, 'it': 8, 'ran': 9,..}






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_dir_train', type=str, default='./data_in/2016klpNER.base_train')
    # parser.add_argument('--data_file_dir_test', type=str, default='./data_in')
    parser.add_argument('--vocab_path', type=str, default='./data_in/vocab_ko_NER.pkl')
    parser.add_argument('--char_vocab_path', type=str, default='./data_in/char_vocab_ko_NER.pkl')
    parser.add_argument('--pos_vocab_path', type=str, default='./data_in/pos_vocab_ko_NER.pkl')
    parser.add_argument('--lex_dict_path', type=str, default='./data_in/lex_dict.pkl')
    parser.add_argument('--threshold', type=int, default=4)
    parser.add_argument('--word2vec_dim', type=int, default=50)

    args = parser.parse_args()
    main(args)


