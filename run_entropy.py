import argparse
import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from einops import rearrange


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        '-d',
                        type=str,
                        default='data/',
                        help='data dir')
    parser.add_argument('--source',
                        type=str,
                        choices=[
                            'webtext',
                            'small-117M',
                            'small-117M-k40',
                            'medium-345M',
                            'medium-345M-k40',
                            'large-762M',
                            'large-762M-k40',
                            'xl-1542M',
                            'xl-1542M-k40',
                        ])
    parser.add_argument('--split',
                        type=str,
                        choices=['train', 'test', 'valid'])
    parser.add_argument('--input', '-i', type=str, default='', help='input file; if specified, --source and --split will be ignored')
    parser.add_argument('--output',
                        '-o',
                        type=str,
                        default='',
                        help='output file or dir')
    parser.add_argument(
        '--model',
        type=str,
        default='',
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
        help=
        'if specified, this model will be used for estimating the NLL in replace of the default models'
    )
    return parser

def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)['text'])
    return texts

def load_model(args):
    model_class = GPT2LMHeadModel
    tokenizer_class = GPT2Tokenizer
    if args.source == 'webtext':
        pretrained_weights = 'gpt2'
    elif args.source.startswith('small'):
        pretrained_weights = 'gpt2'
    elif args.source.startswith('medium'):
        pretrained_weights = 'gpt2-medium'
    elif args.source.startswith('large'):
        pretrained_weights = 'gpt2-large'
    elif args.source.startswith('xl'):
        pretrained_weights = 'gpt2-xl'

    if args.model:
        pretrained_weights = args.model

    model = model_class.from_pretrained(pretrained_weights)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model {pretrained_weights} loaded")
    return model, tokenizer


@torch.no_grad()
def process(model, tokenizer, args):
    device = model.device
    criterian = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)

    if len(args.input) > 0:
        data = [line.strip() for line in open(args.input)]
    else:
        data = _load_split('data', source=args.source, split=args.split)

    if len(args.output) > 0:
        output_file = args.output
    elif len(args.input) > 0:
        output_file = os.path.join(
            args.data_dir,
            f'{args.input}.model={args.model}.nll')
    else:
        output_file = os.path.join(
            args.data_dir,
            f'{args.source}.{args.split}.model={args.model}.nll')

    with open(output_file, 'w') as fw:
        for line in tqdm(data):
            encoded_input = tokenizer(line,
                                      max_length=1024,
                                      truncation=True,
                                      return_tensors='pt').to(device)
            input_ids = encoded_input['input_ids']

            output = model(**encoded_input, labels=input_ids)
            logits = output.logits.to(device)
            target = encoded_input['input_ids'].to(device)

            logits = rearrange(logits, 'B L V -> B V L')
            shift_logits = logits[
                ..., :, :-1]  # Use the first L-1 tokens to predict the next
            shift_target = target[..., 1:]

            nll_loss = criterian(log_softmax(shift_logits),
                                 shift_target).squeeze()
            res = nll_loss.tolist()
            if not isinstance(res, list):
                res = [res]

            try:
                res_str = ' '.join(f'{num:.4f}' for num in res)
            except Exception:
                print('line:', line)
                print('input_ids:', input_ids)
                print('logits.shape:', logits.shape)
                print('res:', res)
                raise
            else:
                fw.write(f'{res_str}\n')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    model, tokenizer = load_model(args)
    process(model, tokenizer, args)