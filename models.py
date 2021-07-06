import torch
import torch.nn as nn

from utils import xavier_uniform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        h0 = torch.zeros(1, num_data, self.num_hidden).to(DEVICE)
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
        h0 = torch.zeros(1, num_data, self.num_hidden).to(DEVICE)
        c0 = torch.zeros(1, num_data, self.num_hidden).to(DEVICE)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.linear(output)
        return output


class Layer1(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Layer1, self).__init__()
        self.W_in = nn.Parameter(xavier_uniform(torch.empty(num_inputs, num_outputs)))
        self.b_in = nn.Parameter(torch.zeros(num_outputs,))
        self.W_rec = nn.Parameter(xavier_uniform(torch.empty(num_outputs, num_outputs)))
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
        self.W_in = nn.Parameter(xavier_uniform(torch.empty(num_inputs, num_outputs)))
        self.b_in = nn.Parameter(torch.zeros(num_outputs,))
        self.W_rec = nn.Parameter(xavier_uniform(torch.empty(num_outputs, num_outputs)))
        self.tanh = nn.Tanh()

    def forward(self, x, x_prev, step):
        if step % 2 == 0:
            return self.identity(x_prev)
        a = x @ self.W_in + x_prev @ self.W_rec + self.b_in
        z = self.tanh(a)
        return z


class LayerOut(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LayerOut, self).__init__()
        self.W_out = nn.Parameter(xavier_uniform(torch.empty(num_inputs, num_outputs)))
        self.b_out = nn.Parameter(torch.zeros(num_outputs,))
        self.tanh = nn.Tanh()

    def forward(self, x):
        a = x @ self.W_out + self.b_out
        z = self.tanh(a)
        return z


class LayerOutScalar(LayerOut):
    def forward(self, x):
        a = torch.zeros(x.size(0), 1).to(DEVICE)
        for i in range(x.size(0)):
            for j in range(self.W_out.size(1)):
                for k in range(self.W_out.size(0)):
                    a[i, j] += x[i, k] * self.W_out[k, j]
                a[i, j] += self.b_out[j]
        z = self.tanh(a)
        return z


class AlarmworkRNN(nn.Module):
    def __init__(
            self, num_inputs, num_hidden, num_outputs, batch_size, seq_len,
            scalar=False
    ):
        super(AlarmworkRNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.layer1 = Layer1(num_inputs, num_hidden)
        self.layer2 = Layer2(num_inputs, num_hidden)
        self.layer_out = None
        if scalar:
            self.layer_out = LayerOutScalar(num_hidden, num_outputs)
        else:
            self.layer_out = LayerOut(num_hidden, num_outputs)

    def forward(self, x):
        num_data, max_seq_len, _ = x.shape
        z_layer1 = torch.zeros((num_data, self.num_hidden)).to(DEVICE)
        z_layer2 = torch.zeros((num_data, self.num_hidden)).to(DEVICE)
        for step in range(max_seq_len):
            element = x[:, step, :]
            z_layer1 = self.layer1(element, z_layer1, z_layer2)
            z_layer2 = self.layer2(element, z_layer2, step)
            z_out = self.layer_out(z_layer1)
        return z_out

