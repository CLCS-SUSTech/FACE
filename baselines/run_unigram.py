from collections import Counter


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
    for word, count in freqs.items():
        if count >= min_count:
            word2id[word] = len(word2id)
            id2word[word2id[word]] = word
        else:
            freqs['[UNK]'] += count
    return word2id, id2word, freqs

def main(args):
    data = load_data(args.input)
    word2id, id2word, freqs = build_vocab(data, min_count=args.min_count)
    total_freq = sum(freqs.values())
    
    with open(args.output, 'w') as f:
        for d in data:
            probs = []
            for word in d:
                if word2id[word] != 0:
                    probs.append(freqs[word] / total_freq)
                else:
                    probs.append(freqs['[UNK]'] / total_freq)
            f.write(' '.join(f'{p:.4f}' for p in probs) + '\n')