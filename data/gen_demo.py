import json
import random

def sample_text(jsonl_file, n_samples, remove_newline=True):
    with open(jsonl_file, 'r') as fr:
        all_lines = fr.readlines()
        random.shuffle(all_lines)
        if remove_newline:
            return [json.loads(line)['text'].replace('\n', ' ') for line in all_lines[:n_samples]]
        else:
            return [json.loads(line)['text'] for line in all_lines[:n_samples]]

N = 1000
human_text = []
human_text.extend(sample_text('human/webtext.test.jsonl', N//2))
human_text.extend(sample_text('human/webtext.valid.jsonl', N//2))
with open('demo_human.txt', 'w') as fw:
    fw.write('\n'.join(human_text))

model_text = []
model_text.extend(sample_text('gpt2_origin/small-117M.test.jsonl', N//2))
model_text.extend(sample_text('gpt2_origin/small-117M.valid.jsonl', N//2))
with open('demo_model.txt', 'w') as fw:
    fw.write('\n'.join(model_text))