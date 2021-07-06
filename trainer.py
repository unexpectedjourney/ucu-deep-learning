from utils import (
        read_data_adding_problem_torch, adding_problem_evaluate, get_batches)


class BasicTrainer:
    def __init__(
            self,
            seq_len,
            num_inputs,
            num_hidden,
            num_outputs,
            batch_size,
            model,
            loss_fn,
            optimizer,
    ):
        self.seq_len = seq_len
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.X_train, self.T_train = read_data_adding_problem_torch(
                f'adding_problem_data/adding_problem_T={seq_len}_train.csv'
        )
        self.X_dev, self.T_dev = read_data_adding_problem_torch(
                f'adding_problem_data/adding_problem_T={seq_len}_dev.csv'
        )
        self.X_test, self.T_test = read_data_adding_problem_torch(
                f'adding_problem_data/adding_problem_T={seq_len}_test.csv'
        )

    def train(self):
        for e in range(1, 51):
            self.model.eval()
            dev_acc = adding_problem_evaluate(self.model(self.X_dev), self.T_dev)
            print(f'T = {self.seq_len}, epoch = {e}, DEV accuracy = {dev_acc}%%')
            if dev_acc > 99.5:
                break
            self.model.train()
            for X_batch, T_batch in get_batches(
                    self.X_train, self.T_train, batch_size=self.batch_size
            ):
                Y_batch = self.model(X_batch)
                loss = self.loss_fn(Y_batch, T_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return dev_acc, e

    def eval(self):
        test_acc = adding_problem_evaluate(self.model(self.X_test), self.T_test)
        print(f'\nTEST accuracy = {test_acc}')
        return test_acc

