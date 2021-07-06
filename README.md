# ucu-deep-learning
This homework contains the solution for RNN, LSTM, and AlarmworkRNN realization and comparison.

## Dependencies
This application uses external library __pytorch__, __fire__, __pandas__, so please install it to work with this application.

```bash
pip install -r requirements.txt
```
## Dataset
Firstly, to create the dataset for our task, please use:
```bash
python3 generate_data_adding_problem.py --seq_len=10
python3 generate_data_adding_problem.py --seq_len=50
python3 generate_data_adding_problem.py --seq_len=70
python3 generate_data_adding_problem.py --seq_len=100
```
## Execution
To execute this project with particular model and sequence length, use:
```bash
python3 train_rnn.py --model_name={srnn, lstm, alarmrnn} --seq_len={10, 50, 70, 100}
```
E.g.:
```bash
python3 train_rnn.py --model_name=alarmrnn --seq_len=50
```
To test scalar realization of AlarmworkRNN LayerOut, please, use:
```bash
python3 train_rnn.py --model_name=alarmrnn --seq_len={10, 50, 70, 100} --scalar
```
To get execution time of your model, you can use:
```bash
python3 train_rnn.py --model_name={srnn, lstm, alarmrnn} --seq_len={10, 50, 70, 100} --clock_in
```
To get the whole report of the different combinations of all models and sequence length, please, use:
```bash
python3 train_rnn.py --execute_all
```

## Results of __execute_all__ attribute 
```
Report:
               model  seq_len  epoch  dev_acc  test_acc
0   SimpleRNNFromBox       10     40     99.6      99.8
1   SimpleRNNFromBox       50     50     37.2      36.4
2   SimpleRNNFromBox       70     50     37.3      35.1
3   SimpleRNNFromBox      100     50     34.4      33.5
4          LSTMModel       10     27     99.8     100.0
5          LSTMModel       50     50     36.3      35.6
6          LSTMModel       70     50     37.4      35.3
7          LSTMModel      100     50     34.5      33.3
8       AlarmworkRNN       10     16     99.9      99.8
9       AlarmworkRNN       50     50     97.4      97.6
10      AlarmworkRNN       70     50     96.5      97.7
11      AlarmworkRNN      100     50     97.3      90.2 
```
Hence, as you can see, AlarmworkRNN performs the best.

## Results of Vector and Scalar forms of AlarmworkRNN:
```
Report:
                  model  seq_len  epoch  dev_acc  test_acc  time used
0  AlarmworkRNN(vector)       10     17     99.8      99.8     22.188
1  AlarmworkRNN(scalar)       10     12     99.8      99.8   3220.640
```
So, we can conclude that each model gives the same accuracy value, but the scalar form is much slower than vector one.

