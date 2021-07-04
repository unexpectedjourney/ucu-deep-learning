import fire
import torch
from torch import nn

from trainer import BasicTrainer, AlarmworkTrainer
from models import (SimpleRNNFromBox, AlarmworkRNN, LSTMModel)


print('Welcome to the real world!')


def main(
    seq_len=10,
    num_inputs=2,
    num_hidden=50,
    num_outputs=1,
    batch_size=20,
    model_name="alarmrnn"
):
    loss_fn = nn.MSELoss()
    model = None
    trainer = None

    if model_name == "alarmrnn":
        model = AlarmworkRNN(num_inputs, num_hidden, num_outputs)   
    elif model_name == "lstm":
        model = LSTMModel(num_inputs, num_hidden, num_outputs)
    elif model_name == "srnn":
        model = SimpleRNNFromBox(num_inputs, num_hidden, num_outputs)
    else:
        print("No model")
        return
    print(type(model).__name__)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if model_name == "alarmrnn":
        trainer = AlarmworkTrainer(
            seq_len, num_inputs, num_hidden, num_outputs, batch_size, model,
            loss_fn, optimizer
        )
    else:
        trainer = BasicTrainer(
            seq_len, num_inputs, num_hidden, num_outputs, batch_size, model,
            loss_fn, optimizer
        )
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    fire.Fire(main)

