import argparse
import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch.nn as nn
from einops import rearrange
import numpy as np
from model import Model


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', 
                        help='input file', required=True)
    parser.add_argument('--output','-o', type=str, default='',
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
    parser.add_argument('--model_path', type=str, default='', 
                        help='load custom models specified by the path')
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--start_batch', type=int, default=0)
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
    """
    For pretrained models from Huggingface transformers 
    """
    device = model.device
    print(f'model is on device: {device}')
    criterian = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)

    with open(args.input, 'r') as fr:
        data = [line.strip() for line in fr.readlines()]
    num_batches = len(data) // args.batch_size
    if len(data) % args.batch_size > 0:
        num_batches += 1

    with open(args.output, 'w') as fw:
        for i in tqdm(range(args.start_batch, num_batches)):
            batch = data[i*args.batch_size: (i*args.batch_size+args.batch_size)]
            if len(batch) == 0:
                continue
            try:
                encoded_input = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
            except Exception:
                raise
            input_ids = encoded_input['input_ids']
            mask = encoded_input['attention_mask']

            try:
                output = model(**encoded_input, labels=input_ids)
            except RuntimeError:
                print(f'batch index: {i}')
                # print(f'batch: {batch}')
                print('encoded_input.input_ids: {}'.format(input_ids))
                raise
            logits = output.logits.to(device)
            target = encoded_input['input_ids'].to(device)

            logits = rearrange(logits, 'B L V -> B V L') 
            shift_logits = logits[..., :, :-1] # Use the first L-1 tokens to predict the next
            shift_target = target[..., 1:]
            mask = mask[..., 1:]

            nll_loss = criterian(log_softmax(shift_logits), shift_target).squeeze()
            for i in range(nll_loss.size(0)):
                out = nll_loss[i,:].squeeze()
                out_masked = torch.masked_select(out, mask[i,:]>0)
                res_str = ' '.join(f'{num:.4f}' for num in out_masked)
                fw.write(f'{res_str}\n')


@torch.no_grad()
def process_custom(args):
    """
    For custom models
    """
    model = Model(args.model_path)

    with open(args.input, 'r') as fr:
        data = [line.strip() for line in fr.readlines()]
    num_batches = len(data) // args.batch_size
    if len(data) % args.batch_size > 0:
        num_batches += 1
    
    with open(args.output, 'w') as fw:
        for i in tqdm(range(args.start_batch, num_batches)):
            batch = data[i*args.batch_size: (i*args.batch_size+args.batch_size)]
            if len(batch) == 0:
                continue 
            try:
                res = model.forward_batch(batch)
            except Exception:
                print(f'batch index: {i}')
                # print(f'batch: {batch}')
                raise
            #todo: implement batch nll calculation


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.model_path:
        pass
    else:
        model, tokenizer = load_model(args)
        process(model, tokenizer, args)