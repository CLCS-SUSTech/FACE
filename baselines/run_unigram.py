from collections import Counter
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, required=True)
parser.add_argument('--output', '-o', type=str, required=True)
parser.add_argument('--min_count', type=int, default=3)

def tokenizer(text: str) -> list[str]:
    return text.strip().split()

def load_data(file_name: str) -> list[list[str]]:
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            data.append(tokenizer(line))
    return data

def build_vocab(data: list[list[str]], min_count=3):
    word2id = {'[UNK]': 0}
    id2word = {0: '[UNK]'}
    freqs = Counter()
    for line in data:
        freqs.update(line)
    unk_count = 0
    for word, count in freqs.items():
        if count >= min_count:
            word2id[word] = len(word2id)
            id2word[word2id[word]] = word
        else:
            unk_count += count
    freqs['[UNK]'] = unk_count
    return word2id, id2word, freqs

def main(args):
    data = load_data(args.input)
    word2id, id2word, freqs = build_vocab(data, min_count=args.min_count)
    total_freq = sum(freqs.values())
    
    with open(args.output, 'w') as f:
        for d in data:
            nlls = []
            for word in d:
                if word in word2id:
                    prob = freqs[word] / total_freq
                else:
                    prob = freqs['[UNK]'] / total_freq
                nlls.append(-np.log(prob))
            f.write(' '.join(f'{val:.4f}' for val in nlls) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)