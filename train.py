#-*-coding: utf-8

import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import subprocess
import joblib
import re
from collections import OrderedDict


import os
import argparse
import pickle
from data_utils import Vocabulary

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CNN_BiLSTM import CNNBiLSTM
from data_loader import get_loader
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
# from logger import Logger
from pprint import pprint







def main(args):
    gpu_index = None
    if args.gpu_index != 0:
        gpu_index = args.gpu_index

    def to_np(x):
        return x.data.cpu().numpy()

    def to_var(x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda(gpu_index)
        return Variable(x, volatile=volatile)

    f_result = open(args.data_file_dir_logs, 'w')
    f_result.write(str(args.data_file_dir_train))
    f_result.write('\n')

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
    # cnn_bilstm_tagger.load_state_dict(torch.load(args.model_load_path, map_location=lambda storage, loc: storage))

    # create model directory
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if torch.cuda.is_available():
        cnn_bilstm_tagger.cuda(gpu_index)

    data_loader = get_loader(data_file_dir=args.data_file_dir_train,
                             vocab=vocab,
                             char_vocab=char_vocab,
                             pos_vocab=pos_vocab,
                             lex_dict=lex_dict,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             dataset='both')

    test_data_loader = get_loader(data_file_dir=args.data_file_dir_test,
                                  vocab=vocab,
                                  char_vocab=char_vocab,
                                  pos_vocab=pos_vocab,
                                  lex_dict=lex_dict,
                                  batch_size=args.test_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)


    # Loss and Optimizer
    learning_rate = args.learning_rate
    momentum = args.momentum

    cnn_bilstm_tagger_parameters = filter(lambda p: p.requires_grad, cnn_bilstm_tagger.parameters())
    optimizer = torch.optim.SGD(cnn_bilstm_tagger_parameters, lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss(ignore_index=0)#nn.NLLLoss() #nn.CrossEntropyLoss()#

    max_macro_f1_score = 0
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for step, (x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, labels, lengths) in enumerate(data_loader):

            # try:
            padded_word_tokens_matrix = to_var(padded_word_tokens_matrix)
            padded_char_tokens_matrix = to_var(padded_char_tokens_matrix)
            padded_pos_tokens_matrix = to_var(padded_pos_tokens_matrix)
            padded_lex_tokens_matrix = to_var(padded_lex_tokens_matrix)
            # padded_lex_tokens_matrix.requires_grad = False
            labels = to_var(labels)


            cnn_bilstm_tagger.zero_grad()
            labels = pack_padded_sequence(labels, lengths, batch_first=True)[0] #[0] -> data, [1] -> batch_size
            predictions = cnn_bilstm_tagger(padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths)
            # features = cnn_bilstm_crf_tagger(padded_word_tokens_matrix, padded_char_tokens_matrix,
            #                                     padded_pos_tokens_matrix, lengths)


            max_labels, argmax_labels = labels.max(1)

            # loss = cnn_bilstm_crf_tagger.loss(features, argmax_labels)
            # viterbi_score, best_tag_sequence = cnn_bilstm_crf_tagger.viterbi_decode(features)
            # argmax_predictions = best_tag_sequence


            max_predictions, argmax_predictions = predictions.max(1)

            # print("predictions",predictions)
            # print("labels",labels)
            loss = criterion(predictions, argmax_labels)
            loss.backward()

            # Update weight parameters
            optimizer.step()


            # Acc
            accuracy = (argmax_labels == argmax_predictions).float().mean() # Different Dim, but it works (batch_size, 1) & (batch_size)

            # print("to_np(argmax_labels):",to_np(argmax_labels))
            # print("to_np(argmax_predictions):", to_np(argmax_predictions))
            # f1_score(to_np(argmax_labels), to_np(argmax_predictions), average='macro')

            print("Training:")
            print("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, accuracy: %.4f, macro-avg f1: %.4f"%
                  (epoch + 1, args.num_epochs, step + 1, total_step, loss.data[0], accuracy.data[0], f1_score(to_np(argmax_labels), to_np(argmax_predictions), average='macro')))
                # f_result.write('\n')
                # f_result.write("Training:")
                # f_result.write('\n')
                # f_result.write("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, accuracy: %.4f"%
                #       (epoch + 1, args.num_epochs, step + 1, total_step, loss.data[0], accuracy.data[0]))
                # f_result.write('\n')

            # except Exception as e: # Cuda out of memory
            #     print("out of memory!, skip this batch")
            #     print(e)
            #     continue

            #
            # # Test
            if (step + 1) % args.test_step == 0:

                cnn_bilstm_tagger.eval()

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

                        if len(argmax_labels.size()) != len(labels.size()):  # Check that class dimension is reduced or not (API version issue, pytorch 0.1.12)
                            max_labels, argmax_labels = labels.max(1, keepdim=True)
                            max_predictions, argmax_predictions = predictions.max(1, keepdim=True)

                        # argmax_labels = argmax_labels.squeeze(1)


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
                if (max_macro_f1_score < macro_f1_score):
                    max_macro_f1_score = macro_f1_score

                print("")
                print("Test:")
                print("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, accuracy: %.4f, F1 Score: %.4f, Max F1 Score: %.4f" %
                      (epoch + 1, args.num_epochs, step + 1, total_step, loss.data[0], accuracy.data[0], macro_f1_score, max_macro_f1_score))
                print("")
                print("classification_report:")
                target_names = ['B_LC','B_DT','B_OG','B_TI','B_PS','I','O','<PAD>','<START>','<STOP>']
                print(classification_report(argmax_labels.cpu().data.numpy(), argmax_predictions.cpu().data.numpy(), target_names=target_names))

                f_result.write('\n')
                f_result.write("Test:")
                f_result.write('\n')
                f_result.write("Epoch [%d/%d], Step [%d/%d], Loss: %.4f, accuracy: %.4f, F1 Score: %.4f, Max F1 Score: %.4f" %
                      (epoch + 1, args.num_epochs, step + 1, total_step, loss.data[0], accuracy.data[0], macro_f1_score, max_macro_f1_score))
                f_result.write("classification_report:")
                f_result.write('\n')
                f_result.write(classification_report(argmax_labels.cpu().data.numpy(), argmax_predictions.cpu().data.numpy(), target_names=target_names))
                f_result.write('\n')

                cnn_bilstm_tagger.train()

            # Save the models
            if (step + 1) % args.save_step == 0:
                torch.save(cnn_bilstm_tagger.state_dict(),
                           os.path.join(args.model_path,
                                        'cnn_bilstm_crf_tagger-%d-%d_f1_%.4f_maxf1_%.4f_%d_%d.pkl' % (
                                        epoch + 1, step + 1, macro_f1_score, max_macro_f1_score, args.embed_size, args.hidden_size)))


    f_result.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file_dir_train', type=str, default='./data_in/2016klpNER.base_train')
    parser.add_argument('--data_file_dir_test', type=str, default='./data_in/2016klpNER.base_test')
    parser.add_argument('--data_file_dir_logs', type=str, default='./data_out/results.txt')
    parser.add_argument('--vocab_path', type=str, default='./data_in/vocab_ko_NER.pkl')
    parser.add_argument('--char_vocab_path', type=str, default='./data_in/char_vocab_ko_NER.pkl')
    parser.add_argument('--pos_vocab_path', type=str, default='./data_in/pos_vocab_ko_NER.pkl')
    parser.add_argument('--lex_dict_path', type=str, default='./data_in/lex_dict.pkl')
    parser.add_argument('--model_load_path', type=str, default='./data_in/cnn_bilstm_crf_tagger-50-52.pkl')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=3) #64
    parser.add_argument('--test_batch_size', type=int, default=30)  # 64
    parser.add_argument('--embed_size', type=int, default=100) #200
    parser.add_argument('--hidden_size', type=int, default=100)

    parser.add_argument('--learning_rate', type=int, default=1e-1)
    parser.add_argument('--momentum', type=int, default=0.6)

    parser.add_argument('--test_step', type=int, default=300)
    parser.add_argument('--save_step', type=int, default=300)
    parser.add_argument('--model_path', type=str, default='./data_out')
    parser.add_argument('--gpu_index', type=int, default=0)

    args = parser.parse_args()
    main(args)