import argparse
import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from einops import rearrange
from config import load_config
from model import Model


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', 
                        help='input file', required=True)
    parser.add_argument('--output',
                        '-o',
                        type=str,
                        default='',
                        help='output file', required=True)
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
        help=
        'if specified, this model will be used for estimating the entropy \
            (negative log-likelihood output) in replace of the default models'
    )
    parser.add_argument('--model_path', type=str, default='', help='load model locally if specified')

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to the configuration file"
    )
    return parser


def load_model(args):
    if len(args.model_path)>0:
        model_path = args.model_path
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer(tokenizer_file=os.path.join(model_path, 'tokenizer.json'),
                                  vocab_file=os.path.join(model_path, 'vocab.json'),
                                  merges_file=os.path.join(model_path, 'merges.txt'))
    else:
        model_path = args.model
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def process(model, tokenizer, args):
    device = model.device
    criterian = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)

    with open(args.input, 'r') as fr:
        data = [line.strip() for line in fr.readlines()]
    with open(args.output, 'w') as fw:
        for line in tqdm(data):
            encoded_input = tokenizer(line,
                                      max_length=1024,
                                      truncation=True,
                                      return_tensors='pt').to(device)
            input_ids = encoded_input['input_ids']

            try:
                output = model(**encoded_input, labels=input_ids)
            except Exception:
                print('line:', line)
                print('input_ids:', input_ids)
                raise
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


@torch.no_grad()
def process_custom(args):
    """
    For custom models specified in configuration file
    """
    # Load model and data
    model = Model(args.model_est)
    with open(args.input, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    # Compute
    results = []
    for line in tqdm(data):
        logits, nlls = model.forward(line)
        results.append(nlls)
    # Write results
    with open(args.output, 'w') as f:
        for res in results:
            if isinstance(res, torch.Tensor):
                res = res.numpy().tolist()
            res_str = ' '.join(f'{num:.4f}' for num in res)
            f.write(f'{res_str}\n')


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.config is not None:
        args = load_config(args)
        process_custom(args)
    else:
        model, tokenizer = load_model(args)
        process(model, tokenizer, args)