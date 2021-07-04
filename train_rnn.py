import pathlib
import torch
from torch import nn


SEQ_LEN = 10
NUM_INPUTS = 2
NUM_HIDDEN = 50
NUM_OUTPUTS = 1
BATCH_SIZE = 20


def read_data_adding_problem(csv_filename):
    lines = pathlib.Path(csv_filename).read_text().splitlines()
    values, markers, adding_results = [], [], []
    cnt = 0
    for line in lines:
        cnt += 1
        if cnt % 3 == 1:
            curr_values = [float(s) for s in line.split(',')]
            values.append(curr_values)
        elif cnt % 3 == 2:
            curr_markers = [float(s) for s in line.split(',')]
            markers.append(curr_markers)
        else:
            curr_adding_result = float(line.split(',')[0])
            adding_results.append(curr_adding_result)
    return values, markers, adding_results


def read_data_adding_problem_torch(csv_filename):
    values, markers, adding_results = read_data_adding_problem(csv_filename)
    assert len(values) == len(markers) == len(adding_results)
    num_data = len(values)
    seq_len = len(values[0])
    X = torch.Tensor(num_data, seq_len, 2)
    T = torch.Tensor(num_data, 1)
    for k, (curr_values, curr_markers, curr_adding_result) in \
            enumerate(zip(values, markers, adding_results)):
        T[k] = curr_adding_result
        for n, (v, m) in enumerate(zip(curr_values, curr_markers)):
            X[k, n, 0] = v
            X[k, n, 1] = m
    return X, T


def get_batches(X, T, batch_size):
    num_data, max_seq_len, _ = X.shape
    for idx1 in range(0, num_data, batch_size):
        idx2 = min(idx1 + batch_size, num_data)
        yield X[idx1:idx2, :, :], T[idx1:idx2, :]


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


class AlarmworkRNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(AlarmworkRNN, self).__init__()
        self.W_in = nn.Parameter(...)
        # add your code here
        #
        #

    def forward(self, x):
        pass
        # add your code here
        #
        #


class LSTMModel(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_inputs, num_hidden)
        self.linear = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


def adding_problem_evaluate(outputs, gt_outputs):
    assert outputs.shape == gt_outputs.shape
    num_data = outputs.shape[0]
    num_correct = 0
    for i in range(num_data):
        y = outputs[i].item()
        t = gt_outputs[i].item()
        if abs(y - t) < 0.1:
            num_correct += 1
    acc = num_correct*100 / len(outputs)
    return acc


print('Welcome to the real world!')

X_train, T_train = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_train.csv' % SEQ_LEN)
X_dev, T_dev = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_dev.csv' % SEQ_LEN)
X_test, T_test = read_data_adding_problem_torch('adding_problem_data/adding_problem_T=%03d_test.csv' % SEQ_LEN)

# model = SimpleRNNFromBox(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
model = AlarmworkRNN(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
# model = LSTMModel(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
print(type(model).__name__)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for e in range(50):
    model.eval()
    dev_acc = adding_problem_evaluate(model(X_dev), T_dev)
    print(f'T = {SEQ_LEN}, epoch = {e}, DEV accuracy = {dev_acc}%%')
    if dev_acc > 99.5:
        break
    model.train()
    for X_batch, T_batch in get_batches(X_train, T_train, batch_size=BATCH_SIZE):
        Y_batch = model(X_batch)
        loss = loss_fn(Y_batch, T_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_acc = adding_problem_evaluate(model(X_test), T_test)
print(f'\nTEST accuracy = {test_acc}')
