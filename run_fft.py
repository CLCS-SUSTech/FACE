from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from typing import Union
import numpy as np
import pandas as pd
import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='', help='input file')
parser.add_argument('--output', '-o', type=str, default='', help='output file or dir')
parser.add_argument('--method', type=str, default='fft', choices=['fft', 'periodogram'])
parser.add_argument('--preprocess', '-p', type=str, default='none', choices=['none', 'zscore', 'minmax', 'log', 'logzs'])
parser.add_argument('--value', type=str, default='norm', choices=['norm', 'real', 'imag'])
parser.add_argument('--require_sid', action='store_true', default=True, help='if true, append sequence id to output file')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
parser.add_argument('--demo', action='store_true', help='run demo')


# FFT Processor class
class FFTProcessor(object):
    def __init__(self, method, preprocess, value, require_sid, verbose=False):
        self.method = method
        self.preprocess = preprocess
        self.value = value
        self.require_sid = require_sid
        self.verbose = verbose
    
    def _read_data(self, data_file: str, N: int = np.inf):
        data = []
        with open(data_file, 'r') as f:
            count = 0
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                num = list(map(float, line.split()))
                data.append(num)
                count += 1
                if count >= N:
                    break
        return data
    
    def _preprocess(self, input_data: list):
        data = input_data.copy()
        if self.preprocess == 'zscore':
            data_zs = []
            epsion = 1e-6
            for d in data:
                d = np.asarray(d)
                d_mean = np.mean(d)
                d_std = np.std(d)
                d_norm = (d - d_mean) / (d_std + epsion)
                data_zs.append(d_norm)
            data = data_zs.copy()
        elif self.preprocess == 'minmax':
            data_mm = []
            for d in data:
                d = np.asarray(d)
                d_min = np.min(d)
                d_max = np.max(d)
                d_norm = (d - d_min) / (d_max - d_min)
                data_mm.append(d_norm)
            data = data_mm.copy()
        elif self.preprocess == 'log':
            data_log = []
            for d in data:
                d = np.asarray(d)
                d_log = np.log(d + 1)
                data_log.append(d_log)
            data = data_log.copy()
        elif self.preprocess == 'logzs':
            data_logzs = []
            epsion = 1e-6
            for d in data:
                d = np.asarray(d)
                d_log = np.log(d + 1)
                d_mean = np.mean(d_log)
                d_std = np.std(d_log)
                d_norm = (d_log - d_mean) / (d_std + epsion)
                data_logzs.append(d_norm)
            data = data_logzs.copy()
        elif self.preprocess != 'none':
            raise ValueError(f'Unknown preprocess method: {self.preprocess}. Please choose from [none, zscore, minmax, log, logzs].')
        return data
    
    def _create_input_df(self, data: list, require_sid=True):
        if require_sid:
            df = pd.DataFrame({
                'value': np.concatenate(data),
                'sid': np.concatenate([np.repeat(i, len(d)) for i, d in enumerate(data)])
            })
        else:
            df = pd.DataFrame({
                'value': np.concatenate(data)
            })
        return df

    def _periodogram_batch(self, data: list[np.ndarray], require_sid=False):
        """
        Periodogram method (with smoothing window)
        """
        freqs, powers, seq_ids = [], [], []
        for i in tqdm.tqdm(range(len(data))):
            f, p = self._periodogram(data[i])
            freqs.append(f)
            powers.append(p)
            if require_sid:
                seq_ids.append(np.array([i] * len(f)))
        return freqs, powers, seq_ids
    
    def _periodogram(self, data: np.ndarray):
        f, p = signal.periodogram(data)
        return f, p
    
    def _fft_batch(self, data: list, require_sid=False, verbose=False):
        """
        FFT batch
        """
        freqs, powers = [], []
        sids = [] if require_sid else None
        for i in tqdm.tqdm(range(len(data)), disable = not verbose):
            x = data[i]
            try:
                f, p = self._fft(x)
            except Exception:
                print(f'Error in sample {i}: {x}')
                raise
            freqs.append(f)
            powers.append(p)
            if require_sid:
                sids.append(np.array([i] * len(f)))
        return freqs, powers, sids

    def _fft(self, data: Union[np.ndarray, list]):
        """
        FFT
        """
        if isinstance(data, list):
            data = np.asarray(data)
        N = data.shape[-1]
        freq_x = fftshift(fftfreq(N))
        fft_res = fftshift(fft(data))
        if self.value == 'real':
            sp_x = fft_res.real
        elif self.value == 'imag':
            sp_x = fft_res.imag
        else:
            sp_x = np.abs(fft_res) # equivalent to np.sqrt(fft_res.real**2 + fft_res.imag**2)
        return freq_x[len(freq_x)//2:], sp_x[len(sp_x)//2:]
    
    def _create_fft_df(self, freqs, powers, sids=None):
        if sids is not None:
            df = pd.DataFrame.from_dict({
                'sid': np.concatenate(sids),
                'freq': np.concatenate(freqs),
                'power': np.concatenate(powers)
            })
        else:
            df = pd.DataFrame.from_dict({
                'freq': np.concatenate(freqs),
                'power': np.concatenate(powers)
            })
        return df
    
    def process(self, input_data: Union[str, list]):
        """
        Carry out FFT analysis on data stored in input_file
        """
        if isinstance(input_data, str):
            data_list = self._read_data(input_data)
            data = [np.asarray(d) for d in data_list]
        else:
            data = input_data.copy()

        # Preprocess
        data = self._preprocess(data)

        # Compute
        if self.method == 'periodogram':
            freqs, powers, sids = self._periodogram_batch(data, require_sid=self.require_sid, verbose=self.verbose)
        elif self.method == 'fft':
            freqs, powers, sids = self._fft_batch(data, require_sid=self.require_sid, verbose=self.verbose)

        # Collect result 
        df = self._create_fft_df(freqs, powers, sids)

        return df


def demo():
    fft_processor = FFTProcessor(method='fft', preprocess='none', value='real', require_sid=False)
    data_dir = 'data/'
    input_files = ['demo_human.nll.txt', 'demo_model.nll.txt']
    for input_file in input_files:
        df = fft_processor.process(os.path.join(data_dir, input_file))
        output_file = data_dir + input_file[:-4] + '.fft.txt'
        df.to_csv(output_file, index=False)


def main(args):
    fft_processor = FFTProcessor(method=args.method, 
                                 preprocess=args.preprocess, 
                                 value=args.value, 
                                 require_sid=args.require_sid,
                                 verbose=args.verbose)
    df = fft_processor.process(args.input)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        main(args)