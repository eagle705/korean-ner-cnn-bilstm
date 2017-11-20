import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class CNNBiLSTM(nn.Module):
    def __init__(self,  vocab_size, char_vocab_size, pos_vocab_size, lex_ner_size, hidden_size, num_layers, embed_size, word2vec, num_classes):#kernel_num=128, kernel_sizes=[2,3,4],
        super(CNNBiLSTM, self).__init__()

        kernel_size = 1
        channel_input_word = 1  # 2
        channel_input_lexicon = 1  # 2
        kernel_num = 128
        kernel_sizes = [2, 3, 4, 5]
        channel_output = kernel_num

        if word2vec is not None:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
            self.embed.weight = torch.nn.parameter.Parameter(torch.Tensor(word2vec))
            self.embed.weight.requires_grad = False

            self.trainable_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.trainable_embed.weight = torch.nn.parameter.Parameter(torch.Tensor(word2vec))

            self.lstm = nn.LSTM((channel_output * len(kernel_sizes) + 2*embed_size + embed_size + lex_ner_size),
                                hidden_size, num_layers, dropout=0.6, batch_first=True, bidirectional=True)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.lstm = nn.LSTM((channel_output * len(kernel_sizes) + embed_size + embed_size + lex_ner_size),
                                hidden_size, num_layers, dropout=0.6, batch_first=True, bidirectional=True)

        self.char_embed = nn.Embedding(char_vocab_size, embed_size, padding_idx=0)

        self.pos_embed = nn.Embedding(pos_vocab_size, embed_size, padding_idx=0)



        self.convs1 = nn.ModuleList([nn.Conv2d(channel_input_word, channel_output, (kernel_size, embed_size)) for kernel_size in kernel_sizes])
        # self.dropout = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.5)





        self.fc1 = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x, x_char, x_pos, x_lex_embedding, lengths):

        x_word_embedding = self.embed(x)  # (batch,words,word_embedding)
        trainable_x_word_embedding = self.trainable_embed(x)

        char_output = []
        for i in range(x_char.size(1)):
            x_char_embedding = self.char_embed(x_char[:,i]).unsqueeze(1) # (batch,channel_input,words,word_embedding)

            h_convs1 = [F.relu(conv(x_char_embedding)).squeeze(3) for conv in self.convs1]
            h_pools1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h_convs1]  # [(batch,channel_out), ...]*len(kernel_sizes)
            h_pools1 = torch.cat(h_pools1, 1)  # 리스트에 있는걸 쭉 쌓아서 Tensor로 만듬!
            h_pools1 = self.dropout(h_pools1)  # (N,len(Ks)*Co)
            out = h_pools1.unsqueeze(1)  # 단어단위 고려
            char_output.append(out)
            # print("out:",out)
        char_output = torch.cat(char_output, 1) # 단어 단위끼리 붙이고 # torch.cat((h_pools1, h_lexicon_pools1), 1)


        x_pos_embedding = self.pos_embed(x_pos)
        # print("char_output:",char_output)
        # print("x_word_embedding:", x_word_embedding)
        # print("x_pos_embedding:",x_pos_embedding)
        # print("x_lex_embedding:", x_lex_embedding)

        enhanced_embedding = torch.cat((char_output, x_word_embedding, trainable_x_word_embedding, x_pos_embedding), 2) # 임베딩 차원(2)으로 붙이고
        enhanced_embedding = self.dropout(enhanced_embedding)
        enhanced_embedding = torch.cat((enhanced_embedding, x_lex_embedding), 2)
        # enhanced_embedding = torch.cat((char_output, x_word_embedding, x_pos_embedding, x_lex_embedding), 2)  # 임베딩 차원(2)으로 붙이고
        # enhanced_embedding = self.dropout(enhanced_embedding)
        



        packed = pack_padded_sequence(enhanced_embedding, lengths, batch_first=True)
        #packed -> (batch_size * real_length), embedding_dim!! -> it can calculate loss bw/ packed
        # print("lengths",lengths)
        # print("enhanced_embedding.size()",enhanced_embedding.size())
        # print("packed",packed)
        # print("packed",packed)
        # print("pad_packed_sequence(packed, batch_first=True)", pad_packed_sequence(packed, batch_first=True))

        # x_word_embedding = x_word_embedding.unsqueeze(1)  # (batch,channel_input,words,word_embedding)
        # print(packed)
        output_word, state_word = self.lstm(packed)

        # output_word = pad_packed_sequence(output_word, batch_first=True)
        # output_word = self.dropout(output_word[0])
        # output_word = pack_padded_sequence(output_word, lengths, batch_first=True)



        # 초기 CNN 한개 쓰기 위한 코드
        # h_convs1 = [F.relu(conv(x_word_embedding)).squeeze(3) for conv in self.convs1]  # [(batch,channel_out,words), ...]*len(kernel_sizes)
        # h_pools1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h_convs1]  # [(batch,channel_out), ...]*len(kernel_sizes)
        # h_pools1 = torch.cat(h_pools1, 1)  # 리스트에 있는걸 쭉 쌓아서 Tensor로 만듬!
        # h_pools1 = self.dropout(h_pools1)  # (N,len(Ks)*Co)
        # out = h_pools1  # torch.cat((h_pools1, h_lexicon_pools1), 1)

        # print("output_word: ",output_word)


        logit = self.fc1(output_word[0]) #for packed
        # predictions = F.log_softmax(logit)

        return logit

    def sample(self, x, x_char, x_pos, x_lex_embedding, lengths):

        x_word_embedding = self.embed(x)  # (batch,words,word_embedding)
        trainable_x_word_embedding = self.trainable_embed(x)

        char_output = []
        for i in range(x_char.size(1)):
            x_char_embedding = self.char_embed(x_char[:, i]).unsqueeze(1)  # (batch,channel_input,words,word_embedding)

            h_convs1 = [F.relu(conv(x_char_embedding)).squeeze(3) for conv in self.convs1]
            h_pools1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in
                        h_convs1]  # [(batch,channel_out), ...]*len(kernel_sizes)
            h_pools1 = torch.cat(h_pools1, 1)  # 리스트에 있는걸 쭉 쌓아서 Tensor로 만듬!
            h_pools1 = self.dropout(h_pools1)  # (N,len(Ks)*Co)
            out = h_pools1.unsqueeze(1)  # 단어단위 고려
            char_output.append(out)
            # print("out:",out)
        char_output = torch.cat(char_output, 1)  # 단어 단위끼리 붙이고 # torch.cat((h_pools1, h_lexicon_pools1), 1)

        x_pos_embedding = self.pos_embed(x_pos)
        # print("char_output:",char_output)
        # print("x_word_embedding:", x_word_embedding)
        enhanced_embedding = torch.cat((char_output, x_word_embedding, trainable_x_word_embedding, x_pos_embedding), 2)  # 임베딩 차원(2)으로 붙이고
        enhanced_embedding = self.dropout(enhanced_embedding)
        enhanced_embedding = torch.cat((enhanced_embedding, x_lex_embedding), 2)

        output_word, state_word = self.lstm(enhanced_embedding)



        # 초기 CNN 한개 쓰기 위한 코드
        # h_convs1 = [F.relu(conv(x_word_embedding)).squeeze(3) for conv in self.convs1]  # [(batch,channel_out,words), ...]*len(kernel_sizes)
        # h_pools1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h_convs1]  # [(batch,channel_out), ...]*len(kernel_sizes)
        # h_pools1 = torch.cat(h_pools1, 1)  # 리스트에 있는걸 쭉 쌓아서 Tensor로 만듬!
        # h_pools1 = self.dropout(h_pools1)  # (N,len(Ks)*Co)
        # out = h_pools1  # torch.cat((h_pools1, h_lexicon_pools1), 1)

        # print("output_word: ",output_word)
        # print("output_word",output_word.size())
        # print("output_word[0]", output_word[0].size())

        logit = self.fc1(output_word)
        # print("logit", logit)
        # predictions = F.log_softmax(logit)
        # print("predictions", predictions)
        # print("predictions.size()",predictions.size())

        return logit


class EncoderCRF(nn.Module):
    """
    A conditional random field with its features provided by a bidirectional RNN
    (GRU by default). As of right now, the model only accepts a batch size of 1
    to represent model parameter updates as a result of stochastic gradient descent.
    Primarily used for part-of-speech tagging in NLP w/ state-of-the-art results.
    In essence a heavily cleaned up version of the article:
    http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    "Bidirectional LSTM-CRF Models for Sequence Tagging"
    https://arxiv.org/abs/1508.01991
    :param sentence: (seq. length, 1, word embedding size)
    :param sequence (training only): Ground truth sequence label (seq. length)
    :return: Viterbi path decoding score, and sequence.
    """

    def __init__(self, start_tag_index, stop_tag_index, tag_size, embedding_dim, hidden_dim):
        super(EncoderCRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.start_tag_index = start_tag_index
        self.stop_tag_index = stop_tag_index
        self.tag_size = tag_size

        self.encoder = nn.GRU(embedding_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True)

        self.tag_projection = nn.Linear(hidden_dim, self.tag_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size))

        self.hidden = self.init_hidden()

    def to_scalar(self, variable):
        return variable.view(-1).data.tolist()[0]

    def argmax(self, vector, dim=1):
        _, index = torch.max(vector, dim)
        return self.to_scalar(index)

    def state_log_likelihood(self, scores):
        max_score = scores.max()
        max_scores = max_score.unsqueeze(0).expand(*scores.size())
        return max_score + torch.log(torch.sum(torch.exp(scores - max_scores)))

    def init_hidden(self):
        return torch.autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, features):
        energies = torch.Tensor(1, self.tag_size).fill_(-10000.)
        energies[0][self.start_tag_index] = 0.

        energies = torch.autograd.Variable(energies)

        for feature in features:
            best_path = []

            # Forward scores + transition scores + emission scores (based on features)
            next_state_scores = energies.expand(*self.transitions.size()) + self.transitions + feature.unsqueeze(
                0).expand(
                *self.transitions.size())

            for index in range(self.tag_size):
                next_possible_states = next_state_scores[index].unsqueeze(0)
                best_path.append(self.state_log_likelihood(next_possible_states))

            energies = torch.cat(best_path).view(1, -1)

        terminal_energy = energies + self.transitions[self.stop_tag_index]
        return self.state_log_likelihood(terminal_energy)

    def encode(self, sentence):
        self.hidden = self.init_hidden()

        outputs, self.hidden = self.encoder(sentence, self.hidden)
        tag_energies = self.tag_projection(outputs.squeeze())
        return tag_energies

    def _score_sentence(self, features, tags):
        score = torch.autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.start_tag_index]), tags])

        for index, feature in enumerate(features):
            score = score + self.transitions[tags[index + 1], tags[index]] + feature[tags[index + 1]]
        score = score + self.transitions[self.stop_tag_index, tags[-1]]
        return score

    def viterbi_decode(self, features):
        backpointers = []

        energies = torch.Tensor(1, self.tag_size).fill_(-10000.)
        energies[0][self.start_tag_index] = 0

        energies = torch.autograd.Variable(energies)
        for feature in features:
            backtrack = []
            best_path = []

            next_state_scores = energies.expand(*self.transitions.size()) + self.transitions

            for index in range(self.tag_size):
                next_possible_states = next_state_scores[index]
                best_candidate_state = self.argmax(next_possible_states, dim=0)

                backtrack.append(best_candidate_state)
                best_path.append(next_possible_states[best_candidate_state])

            energies = (torch.cat(best_path) + feature).view(1, -1)
            backpointers.append(backtrack)

        # Transition to STOP_TAG.
        terminal_energy = energies + self.transitions[self.stop_tag_index]
        best_candidate_state = self.argmax(terminal_energy)
        path_score = terminal_energy[0][best_candidate_state]

        # Backtrack decoded path.
        best_path = [best_candidate_state]
        for backtrack in reversed(backpointers):
            best_candidate_state = backtrack[best_candidate_state]
            best_path.append(best_candidate_state)

        best_path.reverse()
        best_path = best_path[1:]

        return path_score, best_path

    def loss(self, sentence, tags):
        features = self.encode(sentence)
        forward_score = self._forward_alg(features)
        gold_score = self._score_sentence(features, tags)

        return forward_score - gold_score

    def forward(self, sentence):
        features = self.encode(sentence)

        viterbi_score, best_tag_sequence = self.viterbi_decode(features)
        return viterbi_score, best_tag_sequence