import random
import pathlib

import fire


ROUND_DEC = 2


def generate_seqs(max_len):
    pos1 = random.randint(1, max_len // 2) - 1
    pos2 = random.randint(max_len // 2 + 1, max_len) - 1
    seq1 = []
    for _ in range(max_len):
        x = round((random.random() - 0.5)*0.5, ROUND_DEC)
        seq1.append(x)
    seq2 = [0] * max_len
    seq2[pos1] = 1
    seq2[pos2] = 1
    seq3 = [''] * max_len
    adding_result = seq1[pos1] + seq1[pos2]
    seq3[0] = str(round(adding_result, ROUND_DEC))
    seq1 = [str(x) for x in seq1]
    seq2 = [str(x) for x in seq2]
    s1 = ','.join(seq1)
    s2 = ','.join(seq2)
    s3 = ','.join(seq3)
    return s1, s2, s3


def write_seqs_to_disk(num_samples, csv_filename_out, seq_len):
    lines = []
    for _ in range(num_samples):
        s1, s2, s3 = generate_seqs(seq_len)
        lines.append(s1)
        lines.append(s2)
        lines.append(s3)
    pathlib.Path(csv_filename_out).write_text('\n'.join(lines))
    print(f'\nT = {seq_len}, {len(lines) // 3} samples saved to {csv_filename_out}')


def main(seq_len=10):
    pathlib.Path('adding_problem_data').mkdir(parents=True, exist_ok=True)
    write_seqs_to_disk(
        num_samples=10000,
        csv_filename_out=f'adding_problem_data/adding_problem_T={seq_len}_train.csv', 
        seq_len=seq_len
    )
    write_seqs_to_disk(
        num_samples=1000,
        csv_filename_out=f'adding_problem_data/adding_problem_T={seq_len}_dev.csv', 
        seq_len=seq_len
    )

    write_seqs_to_disk(
        num_samples=1000,
        csv_filename_out=f'adding_problem_data/adding_problem_T={seq_len}_test.csv', 
        seq_len=seq_len
    )


if __name__ == "__main__":
    fire.Fire(main)
