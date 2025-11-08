# Synthesizer

Recreating ["Synthesizer: Rethinking Self-Attention for Transformer Models"](https://arxiv.org/abs/2005.00743) by Tay et al (2021).

## Create environment (3 methods)
```
conda env create -f environments/environment.yml
```
```
conda env create -f environments/gpu_environment.yml
```
```
conda create -n synth python=3.12
conda activate synth
conda install pip
pip install -r environments/requirements.txt
```

## Training
- Run the entire experiment suite: `python main.py`
- Preview the experiment suite: `python main.py --dry-run`
- Run suite subset (manually specify): `python main.py --only wmt14_dense,lm1b_random`

- Checkpoints are added to \<path\>/synthesizer/runs/\<model\>/\<epoch\>

## Evaluation
```
python -m evaluation.rougel_test
python -m evaluation.bleu_test
python -m evaluation.ppl_test
```

## Results
### English -> German Translation
- Seq2Seq (encoder-decoder model)
- Dataset: WMT14-ende
- Evaluation: BLEU
- 7 epochs, 35k warmup, 246575 steps, Noam Scheduler (0.5 scale), GPT-2 tokenizer

### Language Modeling
- CLM (decoder-only model)
- Dataset: LM1B
- Evaluation: PPL
- 5 epochs, 4k warmup, 1183635 steps, LR=5e-4 (cosine scheduler), SP32k tokenizer

### Summarization
- Seq2Seq (encoder-decoder model)
- Dataset: CNN_Dailymail
- Evaluation: ROUGE-L
- 50 epochs, 30k warmup, 112200 steps, Noam Scheduler (0.5 scale), SP32k tokenizer

### Validation Set Scores

|  | BLEU | ROUGE-L | PPL |
|----|----|----|----|
| Vanilla | 27.01 | 19.64 | 38.85 |
| Dense | 23.06 | 15.34 | 35.96 |
| Random | 21.09 | 18.00 | 37.53 |

- dense-vanilla hybrid and random-vanilla hybrid attention models also available
- GLUE, SuperGLUE, PersonaChat, C4, WMT14-enfr datasets also available
- MLM Classification (encoder-only) task also available

