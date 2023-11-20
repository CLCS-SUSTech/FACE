# FACE: Fourier Analysis of Cross-Entropy
Repository for the paper *FACE: Evaluating Natural Language Generation with
Fourier Analysis of Cross-Entropy* (NeurIPS 2023). 

```bibtex
@article{yang2023face,
  title={FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy},
  author={Yang, Zuhao and Yuan, Yingfang and Xu, Yang and Zhan, Shuo and Bai, Huajun and Chen, Kefan},
  journal={arXiv preprint arXiv:2305.10307},
  year={2023}
}
```

## Procedure for using FACE metric
1. Install the required packages in `requirements.txt`.
2. Run `data/gen_demo.py` to generate the demo data: `demo_human.txt` and `demo_model.txt`
```console
$ cd data
$ python gen_demo.py
```
3. Run `run_entropy.py` or `run_entropy_batch.py` to obtain the entropy sequences of the demo data, saved to `demo_human.nll.txt` and `demo_model.nll.txt`.
```console
$ python run_entropy.py --input data/demo_human.txt --output data/demo_human.nll.txt 
```
or
```console
$ python run_entropy_batch.py --input data/demo_model.txt --output data/demo_model.nll.txt --batch_size 8
```
4. Run `run_fft.py` to obtain the spectra of entropy, saved to `demo_human.fft.txt` and `demo_model.fft.txt`.
```console
```
5. Run `run_face.py` to compute the FACE scores.
```console
```