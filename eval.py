#-*-coding: utf-8

import numpy as np
import torch
from torch.autograd import Variable
import os
import argparse
import pickle

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CNN_BiLSTM import CNNBiLSTM
from data_loader import get_loader
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score



def main(args):
    gpu_index = None
    if args.gpu_index != 0:
        gpu_index = args.gpu_index

    def to_np(x):
        return x.data.cpu().numpy()

    def to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)


    # apply word2vec
    from gensim.models import word2vec
    pretrained_word2vec_file = './data_in/word2vec/ko_word2vec_' + str(args.embed_size) + '.model'
    wv_model_ko = word2vec.Word2Vec.load(pretrained_word2vec_file)
    word2vec_matrix = wv_model_ko.wv.syn0


    # build vocab
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("len(vocab): ",len(vocab))
    print("word2vec_matrix: ",np.shape(word2vec_matrix))
    with open(args.char_vocab_path, 'rb') as f:
        char_vocab = pickle.load(f)
    with open(args.pos_vocab_path, 'rb') as f:
        pos_vocab = pickle.load(f)
    with open(args.lex_dict_path, 'rb') as f:
        lex_dict = pickle.load(f)

    NER_idx_dic = {'<unk>': 0, 'LC': 1, 'DT': 2, 'OG': 3, 'TI': 4, 'PS': 5}


    # build models
    cnn_bilstm_tagger = CNNBiLSTM(vocab_size=len(vocab),
                                         char_vocab_size=len(char_vocab),
                                            pos_vocab_size=len(pos_vocab),
                                            lex_ner_size=len(NER_idx_dic),
                                            embed_size=args.embed_size,
                                            hidden_size=args.hidden_size,
                                            num_layers=args.num_layers,
                                            word2vec=word2vec_matrix,
                                            num_classes=10)

    # If you don't use GPU, you can get error here (in the case of loading state dict from Tensor on GPU)
    #  To avoid error, you should use options -> map_location=lambda storage, loc: storage. it will load tensor to CPU
    cnn_bilstm_tagger.load_state_dict(torch.load(args.model_load_path, map_location=lambda storage, loc: storage))

    # create model directory
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if torch.cuda.is_available():
        cnn_bilstm_tagger.cuda(gpu_index)

    # inference mode
    cnn_bilstm_tagger.eval()

    test_data_loader = get_loader(data_file_dir=args.data_file_dir_test,
                                  vocab=vocab,
                                  char_vocab=char_vocab,
                                  pos_vocab=pos_vocab,
                                  lex_dict=lex_dict,
                                  batch_size=args.test_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)


    # Loss and Optimizer
    # learning_rate = args.learning_rate
    # momentum = args.momentum

    # cnn_bilstm_tagger_parameters = filter(lambda p: p.requires_grad, cnn_bilstm_tagger.parameters())
    # optimizer = torch.optim.SGD(cnn_bilstm_tagger_parameters, lr=learning_rate, momentum=momentum)
    # criterion = nn.NLLLoss()


    # # Test
    argmax_labels_list = []
    argmax_predictions_list = []

    for step_test, (x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, labels, lengths) in enumerate(test_data_loader):
        try:
            padded_word_tokens_matrix = to_var(padded_word_tokens_matrix, volatile=True)
            padded_char_tokens_matrix = to_var(padded_char_tokens_matrix, volatile=True)
            padded_pos_tokens_matrix = to_var(padded_pos_tokens_matrix, volatile=True)
            padded_lex_tokens_matrix = to_var(padded_lex_tokens_matrix, volatile=True)
            labels = to_var(labels, volatile=True)


            labels = pack_padded_sequence(labels, lengths, batch_first=True)[0]

            predictions = cnn_bilstm_tagger(padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths)

            max_labels, argmax_labels = labels.max(1)
            max_predictions, argmax_predictions = predictions.max(1)

            if len(argmax_labels.size()) != len(
                    labels.size()):  # Check that class dimension is reduced or not (API version issue, pytorch 0.1.12)
                max_labels, argmax_labels = labels.max(1, keepdim=True)
                max_predictions, argmax_predictions = predictions.max(1, keepdim=True)




            argmax_labels_list.append(argmax_labels)
            argmax_predictions_list.append(argmax_predictions)



        except Exception as e:
            print(e)
            continue

    argmax_labels = torch.cat(argmax_labels_list, 0)
    argmax_predictions = torch.cat(argmax_predictions_list, 0)

    # Acc
    accuracy = (argmax_labels == argmax_predictions).float().mean() #ToDo: Check Dim

    # f1 score
    argmax_labels_np_array = to_np(argmax_labels)
    argmax_predictions_np_array = to_np(argmax_predictions)
    macro_f1_score = f1_score(argmax_labels_np_array, argmax_predictions_np_array, average='macro')


    print("")
    print("Test:")
    print("accuracy: %.4f, F1 Score: %.4f" % (accuracy.data[0], macro_f1_score))
    print("")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file_dir_train', type=str, default='./data_in/2016klpNER.base_train')
    parser.add_argument('--data_file_dir_test', type=str, default='./data_in/2016klpNER.base_test')
    parser.add_argument('--data_file_dir_logs', type=str, default='./data_out/results.txt')
    parser.add_argument('--vocab_path', type=str, default='./data_in/vocab_ko_NER.pkl')
    parser.add_argument('--char_vocab_path', type=str, default='./data_in/char_vocab_ko_NER.pkl')
    parser.add_argument('--pos_vocab_path', type=str, default='./data_in/pos_vocab_ko_NER.pkl')
    parser.add_argument('--lex_dict_path', type=str, default='./data_in/lex_dict.pkl')
    parser.add_argument('--model_load_path', type=str, default='./data_in/cnn_bilstm_tagger-131-200_f1_0.8569_maxf1_0.8569_100_200_2.pkl')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2) #64
    parser.add_argument('--test_batch_size', type=int, default=30)  # 64
    parser.add_argument('--embed_size', type=int, default=100) #50
    parser.add_argument('--hidden_size', type=int, default=200) #100

    parser.add_argument('--learning_rate', type=int, default=1e-1)
    parser.add_argument('--momentum', type=int, default=0.6)


    parser.add_argument('--model_path', type=str, default='./data_out')
    parser.add_argument('--gpu_index', type=int, default=0)

    args = parser.parse_args()
    main(args)