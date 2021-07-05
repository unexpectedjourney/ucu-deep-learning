import torch
import torch.nn as nn


class SimpleRNNFromBox(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SimpleRNNFromBox, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.rnn = nn.RNN(input_size=num_inputs, hidden_size=num_hidden,
                          num_layers=1, batch_first=True, bidirectional=False)
        self.out_layer = nn.Linear(in_features=num_hidden,
                                   out_features=num_outputs)

    def forward(self, X):
        num_data, max_seq_len, _ = X.shape
        h0 = torch.zeros(1, num_data, self.num_hidden)
        output, hn = self.rnn(X, h0) # output.shape: num_data x seq_len x num_hidden
        last_output = output[:, -1, :] # num_data x num_hidden
        Y = self.out_layer(last_output) # num_data x num_outputs
        return Y


class LSTMModel(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(LSTMModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden,
                            num_layers=1, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        num_data, max_seq_len, _ = x.shape
        h0 = torch.zeros(1, num_data, self.num_hidden)
        c0 = torch.zeros(1, num_data, self.num_hidden)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.linear(output)
        return output


class Layer1(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Layer1, self).__init__()
        self.W_in = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_inputs, num_outputs)))
        self.b_in = nn.Parameter(torch.zeros(num_outputs,))
        self.W_rec = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_outputs, num_outputs)))
        self.tanh = nn.Tanh()

    def forward(self, x, x_layer1, x_layer2):
        z12 = x_layer1 + x_layer2
        a = x @ self.W_in + z12 @ self.W_rec + self.b_in
        z = self.tanh(a)
        return z


class Layer2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Layer2, self).__init__()
        self.identity = nn.Identity()
        self.W_in = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_inputs, num_outputs)))
        self.b_in = nn.Parameter(torch.zeros(num_outputs,))
        self.W_rec = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_outputs, num_outputs)))
        self.tanh = nn.Tanh()

    def forward(self, x, x_prev, step):
        if step % 2 == 1:
            return self.identity(x_prev)
        a = x @ self.W_in + x_prev @ self.W_rec + self.b_in
        z = self.tanh(a)
        return z


class LayerOut(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LayerOut, self).__init__()
        self.W_out = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_inputs, num_outputs)))
        self.b_out = nn.Parameter(torch.zeros(num_outputs,))
        self.tanh = nn.Tanh()

    def forward(self, x):
        a = x @ self.W_out + self.b_out
        z = self.tanh(a)
        return z


class AlarmworkRNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, batch_size, seq_len):
        super(AlarmworkRNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.layer1 = Layer1(num_inputs, num_hidden)
        self.layer2 = Layer2(num_inputs, num_hidden)
        self.layer_out = LayerOut(num_hidden, num_outputs)

    def forward(self, x):
        num_data, max_seq_len, _ = x.shape
        z_layer1 = torch.zeros((max_seq_len, self.num_hidden))
        z_layer2 = torch.zeros((max_seq_len, self.num_hidden))
        out = []
        for step, x_seq in enumerate(x):
            z_layer1 = self.layer1(x_seq, z_layer1, z_layer2)
            z_layer2 = self.layer2(x_seq, z_layer2, step)
            z_out = self.layer_out(z_layer1)
            z_out = z_out[-1, :]
            out.append(z_out)
        out = torch.stack(out)
        return out 

