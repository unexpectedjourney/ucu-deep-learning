import time

import fire
import torch
from torch import nn

from trainer import BasicTrainer
from models import (SimpleRNNFromBox, AlarmworkRNN, LSTMModel)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Welcome to the real world!')


def main(
    seq_len=10,
    num_inputs=2,
    num_hidden=50,
    num_outputs=1,
    batch_size=20,
    model_name="alarmrnn",
    scalar=False,
    clock_in=False
):
    loss_fn = nn.MSELoss().to(DEVICE)
    model = None
    trainer = None

    if model_name == "alarmrnn":
        model = AlarmworkRNN(
            num_inputs, num_hidden, num_outputs, batch_size, seq_len, scalar
        )
    elif model_name == "lstm":
        model = LSTMModel(num_inputs, num_hidden, num_outputs)
    elif model_name == "srnn":
        model = SimpleRNNFromBox(num_inputs, num_hidden, num_outputs)
    else:
        print("No model")
        return
    print(type(model).__name__)

    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = BasicTrainer(
        seq_len, num_inputs, num_hidden, num_outputs, batch_size, model,
        loss_fn, optimizer
    )
    if clock_in:
        start_at = time.time()

    dev_acc = trainer.train()
    test_acc = trainer.eval()
    
    if clock_in:
        end_at = time.time()
        print(f"Time used for {model_name} with seq_len={seq_len}: {end_at - start_at}")


if __name__ == "__main__":
    fire.Fire(main)

