import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNBiLSTM(nn.Module):
    def __init__(self,  vocab_size, char_vocab_size, pos_vocab_size, lex_ner_size, hidden_size, num_layers, embed_size, word2vec, num_classes):#kernel_num=128, kernel_sizes=[2,3,4],
        super(CNNBiLSTM, self).__init__()

        kernel_size = 1
        channel_input_word = 1
        channel_input_lexicon = 1 
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

        enhanced_embedding = torch.cat((char_output, x_word_embedding, trainable_x_word_embedding, x_pos_embedding), 2) # 임베딩 차원(2)으로 붙이고
        enhanced_embedding = self.dropout(enhanced_embedding)
        enhanced_embedding = torch.cat((enhanced_embedding, x_lex_embedding), 2)





        packed = pack_padded_sequence(enhanced_embedding, lengths, batch_first=True)
        #packed -> (batch_size * real_length), embedding_dim!! -> it can calculate loss bw/ packed

        output_word, state_word = self.lstm(packed)



        logit = self.fc1(output_word[0]) #for packed

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

        char_output = torch.cat(char_output, 1)  # 단어 단위끼리 붙이고 # torch.cat((h_pools1, h_lexicon_pools1), 1)

        x_pos_embedding = self.pos_embed(x_pos)

        enhanced_embedding = torch.cat((char_output, x_word_embedding, trainable_x_word_embedding, x_pos_embedding), 2)  # 임베딩 차원(2)으로 붙이고
        enhanced_embedding = self.dropout(enhanced_embedding)
        enhanced_embedding = torch.cat((enhanced_embedding, x_lex_embedding), 2)

        output_word, state_word = self.lstm(enhanced_embedding)
        logit = self.fc1(output_word)

        return logit
