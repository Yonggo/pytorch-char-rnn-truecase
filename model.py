import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, dropout, bi_direction=False):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.bi_direction = bi_direction

        self.lstm = nn.LSTM(input_size, rnn_size, num_layers, dropout=dropout, bidirectional=bi_direction)
        self.dropout = nn.Dropout(dropout)
        d = 2 if self.bi_direction is True else 1
        self.fc = nn.Linear(rnn_size * d, input_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input, hidden):
        input = F.one_hot(input, self.input_size).float()
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        if hasattr(self, "bi_direction") is not True:
            self.bi_direction = False
        d = 2 if self.bi_direction is True else 1
        return (torch.zeros(self.num_layers * d, batch_size, self.rnn_size, device=device),
                torch.zeros(self.num_layers * d, batch_size, self.rnn_size, device=device))
