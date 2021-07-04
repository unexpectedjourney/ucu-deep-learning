import torch
from torch import nn

from trainer import BasicTrainer
from models import (SimpleRNNFromBox, AlarmworkRNN, LSTMModel)


SEQ_LEN = 10
NUM_INPUTS = 2
NUM_HIDDEN = 50
NUM_OUTPUTS = 1
BATCH_SIZE = 20

print('Welcome to the real world!')

# model = SimpleRNNFromBox(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
# model = AlarmworkRNN(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
model = LSTMModel(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
print(type(model).__name__)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

trainer = BasicTrainer(
        SEQ_LEN, NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS, BATCH_SIZE, model, 
        loss_fn, optimizer
)
trainer.train()
trainer.eval()

