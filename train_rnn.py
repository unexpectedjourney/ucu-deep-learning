import time

import fire
import pandas as pd
import torch
from torch import nn

from trainer import BasicTrainer
from models import (SimpleRNNFromBox, AlarmworkRNN, LSTMModel)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Welcome to the real world!')


def make_report(
    models_seq,
    num_inputs,
    num_hidden,
    num_outputs,
    batch_size,
    scalar,
    clock_in,
):
    df = pd.DataFrame()

    for model_name, seq_len in models_seq:
        model = None

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

        loss_fn = nn.MSELoss().to(DEVICE)
        model = model.to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        trainer = BasicTrainer(
            seq_len, num_inputs, num_hidden, num_outputs, batch_size, model,
            loss_fn, optimizer
        )
        if clock_in:
            start_at = time.time()

        dev_acc, epoch = trainer.train()
        test_acc = trainer.eval()
        
        if clock_in:
            end_at = time.time()
            print(f"Time used for {model_name} with seq_len={seq_len}: {end_at - start_at}")
        df = df.append(pd.DataFrame({
            "model": [type(model).__name__],
            "seq_len": [seq_len],
            "epoch": [epoch],
            "dev_acc": [dev_acc],
            "test_acc": [test_acc]
        }), ignore_index=True)
    print("Report:")
    print(df)


def main(
    seq_len=10,
    num_inputs=2,
    num_hidden=50,
    num_outputs=1,
    batch_size=20,
    model_name="alarmrnn",
    scalar=False,
    clock_in=False,
    execute_all=False
):
    models_seq = []
    if execute_all:
        for model_name in ("srnn", "lstm", "alarmrnn"):
            for seq_len in (10, 50, 70, 100):
                models_seq.append([model_name, seq_len])
    else:
        models_seq.append([model_name, seq_len])

    make_report(
        models_seq,
        num_inputs,
        num_hidden,
        num_outputs,
        batch_size,
        scalar,
        clock_in,
    )


if __name__ == "__main__":
    fire.Fire(main)

