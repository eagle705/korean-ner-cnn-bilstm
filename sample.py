#-*-coding: utf-8

import numpy as np
import torch

from torch.autograd import Variable
import copy
import os
import argparse
import pickle
from data_utils import Vocabulary


from CNN_BiLSTM import CNNBiLSTM
from data_loader import get_loader
from sklearn.metrics import f1_score

predict_NER_dict = {0: '<PAD>',
                        1: '<START>',
                        2: '<STOP>',
                        3: 'B_LC',
                        4: 'B_DT',
                        5: 'B_OG',
                        6: 'B_TI',
                        7: 'B_PS',
                        8: 'I',
                        9: 'O'}

def parsing_seq2NER(argmax_predictions, x_text_batch):
    predict_NER_list = []
    predict_text_NER_result_batch = copy.deepcopy(x_text_batch[0])  # tuple ([],) -> return first list (batch_size == 1)
    for argmax_prediction_seq in argmax_predictions:
        predict_NER = []
        NER_B_flag = None  # stop B
        prev_NER_token = None
        for i, argmax_prediction in enumerate(argmax_prediction_seq):
            now_NER_token = predict_NER_dict[argmax_prediction.cpu().data.numpy()[0]]
            predict_NER.append(now_NER_token)
            if now_NER_token in ['B_LC', 'B_DT', 'B_OG', 'B_TI', 'B_PS'] and NER_B_flag is None:  # O B_LC
                NER_B_flag = now_NER_token  # start B
                predict_text_NER_result_batch[i] = '<' + predict_text_NER_result_batch[i]
                prev_NER_token = now_NER_token
                if i == len(argmax_prediction_seq) - 1:
                    predict_text_NER_result_batch[i] = predict_text_NER_result_batch[i] + ':' + now_NER_token[-2:] + '>'

            elif now_NER_token in ['B_LC', 'B_DT', 'B_OG', 'B_TI', 'B_PS'] and NER_B_flag is not None:  # O B_LC B_DT
                predict_text_NER_result_batch[i - 1] = predict_text_NER_result_batch[i - 1] + ':' + prev_NER_token[
                                                                                                    -2:] + '>'
                predict_text_NER_result_batch[i] = '<' + predict_text_NER_result_batch[i]
                prev_NER_token = now_NER_token
                if i == len(argmax_prediction_seq) - 1:
                    predict_text_NER_result_batch[i] = predict_text_NER_result_batch[i] + ':' + now_NER_token[-2:] + '>'

            elif now_NER_token in ['I'] and NER_B_flag is not None:
                if i == len(argmax_prediction_seq) - 1:
                    predict_text_NER_result_batch[i] = predict_text_NER_result_batch[i] + ':' + NER_B_flag[-2:] + '>'

            elif now_NER_token in ['O'] and NER_B_flag is not None:  # O B_LC I O
                predict_text_NER_result_batch[i - 1] = predict_text_NER_result_batch[i - 1] + ':' + prev_NER_token[
                                                                                                    -2:] + '>'
                NER_B_flag = None  # stop B
                prev_NER_token = now_NER_token

                # predict_NER_list.append(predict_NER)
        predict_NER_list.append(predict_NER)
    return predict_NER_list, predict_text_NER_result_batch

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
                                            embed_size=args.embed_size,
                                            lex_ner_size=len(NER_idx_dic),
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
                                  num_workers=args.num_workers,
                                  dataset='klp')


    # Loss and Optimizer
    # learning_rate = args.learning_rate
    # momentum = args.momentum

    # cnn_bilstm_tagger_parameters = filter(lambda p: p.requires_grad, cnn_bilstm_tagger.parameters())
    # optimizer = torch.optim.SGD(cnn_bilstm_tagger_parameters, lr=learning_rate, momentum=momentum)
    # criterion = nn.NLLLoss()


    # Test
    argmax_labels_list = []
    argmax_predictions_list = []

    for step_test, (x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, labels, lengths) in enumerate(test_data_loader):
        try:
            padded_word_tokens_matrix = to_var(padded_word_tokens_matrix, volatile=True)
            padded_char_tokens_matrix = to_var(padded_char_tokens_matrix, volatile=True)
            padded_pos_tokens_matrix = to_var(padded_pos_tokens_matrix, volatile=True)
            padded_lex_tokens_matrix = to_var(padded_lex_tokens_matrix, volatile=True)
            labels = to_var(labels, volatile=True)

            predictions = cnn_bilstm_tagger.sample(padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths)

            max_labels, argmax_labels = labels.max(2)
            max_predictions, argmax_predictions = predictions.max(2)


            if len(argmax_labels.size()) != len(
                    labels.size()):  # Check that class dimension is reduced or not (API version issue, pytorch 0.1.12)
                max_labels, argmax_labels = labels.max(2, keepdim=True)
                max_predictions, argmax_predictions = predictions.max(2, keepdim=True)



            argmax_labels_list.append(argmax_labels)
            argmax_predictions_list.append(argmax_predictions)


            # print("padded_word_tokens_matrix.size()",padded_word_tokens_matrix.size())
            # print("x_text_batch.len()",len(x_text_batch))
            # print("argmax_labels.size()",argmax_labels.size())
            # print("argmax_predictions.size()",argmax_predictions.size())



            predict_NER_list, predict_text_NER_result_batch = parsing_seq2NER(argmax_predictions, x_text_batch)
            label_NER_list, labl_text_NER_result_batch = parsing_seq2NER(argmax_labels, x_text_batch)


            # print("x_text_batch: ",x_text_batch)
            # print("predict_NER_list: ",predict_NER_list)
            # print("predict_text_NER_result_batch: ",predict_text_NER_result_batch)
            # print("label_NER_list: ",label_NER_list)
            # print("labl_text_NER_result_batch: ",labl_text_NER_result_batch)
            # print("x_split_batch: ",x_split_batch)
            x_text_batch = x_text_batch[0]



            def generate_text_result(text_NER_result_batch, x_split_batch):
                prev_x_split = 0 # same split
                text_string = ''
                for i, x_split in enumerate(x_split_batch[0]):
                    if prev_x_split != x_split:
                        text_string = text_string+' '+text_NER_result_batch[i]
                        prev_x_split = x_split
                    else:
                        text_string = text_string +''+ text_NER_result_batch[i]
                        prev_x_split = x_split
                return text_string

            origin_text_string = generate_text_result(x_text_batch, x_split_batch)
            predict_NER_text_string = generate_text_result(predict_text_NER_result_batch, x_split_batch)
            label_text_string = generate_text_result(labl_text_NER_result_batch, x_split_batch)


            print("origin:  ",origin_text_string)
            print("predict: ",predict_NER_text_string)
            print("True: ",label_text_string)
            print("")


        except Exception as e:
            print(e)
            continue

    argmax_labels = torch.cat(argmax_labels_list, 0)
    argmax_predictions = torch.cat(argmax_predictions_list, 0)

    # Acc
    accuracy = (argmax_labels == argmax_predictions).float().mean()

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
    # parser.add_argument('--data_file_dir_test', type=str, default='./data_in/my_test')
    parser.add_argument('--data_file_dir_logs', type=str, default='./data_out/results.txt')
    parser.add_argument('--vocab_path', type=str, default='./data_in/vocab_ko_NER.pkl')
    parser.add_argument('--char_vocab_path', type=str, default='./data_in/char_vocab_ko_NER.pkl')
    parser.add_argument('--pos_vocab_path', type=str, default='./data_in/pos_vocab_ko_NER.pkl')
    parser.add_argument('--lex_dict_path', type=str, default='./data_in/lex_dict.pkl')
    parser.add_argument('--model_load_path', type=str, default='./data_in/cnn_bilstm_tagger-179-400_f1_0.8739_maxf1_0.8739_100_200_2.pkl')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1) #64
    parser.add_argument('--test_batch_size', type=int, default=1)  # 64
    parser.add_argument('--embed_size', type=int, default=100) #50
    parser.add_argument('--hidden_size', type=int, default=200) #100
    parser.add_argument('--learning_rate', type=int, default=1e-1)
    parser.add_argument('--momentum', type=int, default=0.6)
    parser.add_argument('--model_path', type=str, default='./data_out')
    parser.add_argument('--gpu_index', type=int, default=0)

    args = parser.parse_args()
    main(args)