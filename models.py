import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_processing import CLASSES

model_config = {
    'shallow_cnn':{
            'num_channels':100,
            'kernel_sizes':[3,4,5],
            'dropout':0.5
    },

    'deep_cnn':{
        'dropout':0.5
    },

    'char_cnn':{
        'dropout':0.5
    },

    'lstm':{
        'hidden_dim':64,
        'dropout':0.2
    },

    'bi_lstm':{
        'hidden_dim':64,
        'dropout':0.2
    },

    'deep_bi_gru':{
        'hidden_dim': 64,
        'dropout':0.3,
        'fully_connected_dim':32,
        'num_layers':2
    }
}


class Stacker(nn.Module):

    def __init__(self, input_dim):
        super(Stacker, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, len(CLASSES))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def get_model(model_type, args, pretrained_vectors, base_config):
    if model_type == 'shallow_cnn':
        return CNN_Text(args, pretrained_vectors, base_config)
    elif model_type == 'deep_cnn':
        return TextCNN(args, pretrained_vectors, base_config)
    elif model_type == 'char_cnn':
        return CharCNN(args, pretrained_vectors, base_config)
    elif model_type == 'lstm':
        return LSTMClassifier(args, pretrained_vectors, base_config)
    elif model_type == 'bi_lstm':
        return BiLSTMClassifier(args, pretrained_vectors, base_config)
    elif model_type == 'deep_bi_gru':
        return DeepBiGRU(args, pretrained_vectors, base_config)


class CNN_Text(nn.Module):

    def __init__(self, args, pretrained_vectors, base_config):
        super(CNN_Text, self).__init__()
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = base_config['num_channels']
        Ks = base_config['kernel_sizes']

        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(pretrained_vectors)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(base_config['dropout'])
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x.data.t_()
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class TextCNN(nn.Module):
    def __init__(self, args, pretrained_vectors, base_config):
        super(TextCNN, self).__init__()
        self.drop_rate = args.dropout
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.embed.weight.data.copy_(pretrained_vectors)
        self.conv1 = self.conv_and_pool(args.embed_dim, 64, kernel_size=7)
        self.conv2 = self.conv_and_pool(64, 128, kernel_size=3)
        self.conv3 = self.conv_and_pool(128, 256, kernel_size=3)
        self.conv4 = self.conv_and_pool(256, 512, kernel_size=3)
        self.dropout = nn.Dropout(base_config['dropout'])
        self.fc = nn.Linear(512, args.class_num)

    def conv_and_pool(self, input_filters, output_filters, kernel_size):
        return nn.Sequential(
            nn.Conv1d(input_filters, output_filters,
                      kernel_size=kernel_size, stride=1),
            nn.BatchNorm1d(output_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            )


    def forward(self, x_input):
        x_input.data.t_()

        # Embedding
        x = self.embed(x_input)  # dim: (batch_size, max_seq_len, embedding_size)
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # collapse
        x = x.mean(dim=2)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit


class CharCNN(nn.Module):

    def __init__(self, args, pretrained_vectors):
        super(CharCNN, self).__init__()

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(pretrained_vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(7, D), stride=1),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(7, 256), stride=1),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv6 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1),
            nn.ReLU()
        )

        self.maxpool6 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.final_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, C)

    def forward(self, x):


        x.data.t_()
        x = self.embed(x)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = x.transpose(1, 3)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = x.transpose(1, 3)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = x.transpose(1, 3)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = x.transpose(1, 3)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = x.transpose(1, 3)
        x = self.maxpool5(x)
        # BUG HERE !! CONSECUTIVE MAXPOOLING
        x = x.transpose(1, 3)
        x = F.max_pool2d(x, kernel_size=(x.size()[2], 1))
        x = x.view(x.size(0), -1)
        x = self.final_dropout(x)
        x = self.fc(x)

        return x


class LSTMClassifier(nn.Module):

    def __init__(self, args, pretrained_vectors, base_config):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = base_config['hidden_dim']
        self.batch_size = args.batch_size
        self.use_gpu = args.cuda
        self.vocab_size = args.embed_num
        self.embedding_dim = args.embed_dim
        self.word_embeddings = nn.Embedding(args.embed_num, args.embed_dim)
        self.word_embeddings.weight.data.copy_(pretrained_vectors)
        self.lstm = nn.LSTM(args.embed_dim, self.hidden_dim,
                            dropout=base_config['dropout'])
        self.hidden2label = nn.Linear(self.hidden_dim, args.class_num)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        self.hidden = self.init_hidden()
        sentence.data.t_() # batch_size x seq_len
        embeds = self.word_embeddings(sentence) # batch_size x seq_len x embed_dim
        x = embeds.transpose(0, 1)  #  seq_len x batch_size x embed_dim

        lstm_out, self.hidden = self.lstm(x, self.hidden)   # lstm_out: seq_len x batch_size x hidden_size,
                                                            # self.hidden: (last_hidden_state, last_cell_state)
                                                            #              (1 x batch_size x hidden_size, 1 x batch_size x hidden_szie)
                                                            # note that value_of(lstm_out[-1]) == value_of(self.hidden[0])

        logits  = self.hidden2label(lstm_out[-1])           # lstm_out[-1]: 1 x batch_size x hidden_size
                                                            # logits: batch_size x label_size
        return logits


class BiLSTMClassifier(nn.Module):

    def __init__(self, args, pretrained_vectors, base_config):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = base_config['hidden_dim']
        self.batch_size = args.batch_size
        self.use_gpu = args.cuda
        self.vocab_size = args.embed_num
        self.embedding_dim = args.embed_dim
        self.word_embeddings = nn.Embedding(args.embed_num, args.embed_dim)
        self.word_embeddings.weight.data.copy_(pretrained_vectors)
        self.lstm = nn.LSTM(args.embed_dim, self.hidden_dim//2,
                            dropout=base_config['dropout'], bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, args.class_num)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2).cuda())
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2).cuda())
        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2))
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim//2))
        return (h0, c0)

    def forward(self, sentence):
        self.hidden = self.init_hidden()
        sentence.data.t_()  # batch_size x seq_len
        embeds = self.word_embeddings(sentence)  # batch_size x seq_len x embed_dim
        x = embeds.transpose(0, 1) # seq_len x batch_size x embed_dim
        lstm_out, self.hidden = self.lstm(x, self.hidden)  # lstm_out: seq_len x batch_size x hidden_size,
                                                           # self.hidden: (last_hidden_state, last_cell_state)
                                                           #              (1 x batch_size x hidden_size, 1 x batch_size x hidden_size)
                                                           # note that value_of(lstm_out[-1]) == value_of(self.hidden[0])

        logits = self.hidden2label(lstm_out[-1])  # lstm_out[-1]: 1 x batch_size x hidden_size
                                                  # logits: batch_size x label_size
        return logits


class DeepBiGRU(nn.Module):
    def __init__(self, args, pretrained_vectors, base_config):
        super(DeepBiGRU, self).__init__()
        self.hidden_dim = base_config['hidden_dim']
        self.dropout = base_config['dropout']
        self.num_layers = base_config['num_layers']
        self.batch_size = args.batch_size
        self.use_gpu = args.cuda
        self.vocab_size = args.embed_num
        self.embedding_dim = args.embed_dim
        self.word_embeddings = nn.Embedding(args.embed_num, args.embed_dim)
        self.word_embeddings.weight.data.copy_(pretrained_vectors)
        self.gru = nn.GRU(input_size=args.embed_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)

        self.fully_connected1 = nn.Linear(self.hidden_dim*2, base_config['fully_connected_dim'])
        self.fully_connected2 = nn.Linear(base_config['fully_connected_dim'], args.class_num)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim))
        return h0

    def forward(self, sentence):
        sentence.data.t_()  # batch_size x seq_len
        self.batch_size = len(sentence)
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)  # batch_size x seq_len x embed_dim
        x = embeds.transpose(0, 1)  # seq_len x batch_size x embed_dim
        final_all, self.hidden = self.gru(x, self.hidden)
        pre_logits = self.fully_connected1(final_all[-1])
        pre_logits = F.relu(pre_logits)
        logits = self.fully_connected2(pre_logits)
        return logits