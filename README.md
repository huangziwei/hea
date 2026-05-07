# hea: Linear Models with Python

> hea v. (Cantonese) 
> 1. to kill time; to hang around  
> 2. to do something without putting much care or effort into it.
> 3. to go through the motions; to give a carefree response; to beat around the bush; to treat someone lightly
> 4. to place things casually; to disperse (with an outward motion)
> (from [wikitionary](https://en.wiktionary.org/wiki/hea#Verb)) 

This project started as a laid-back exercise of implementing R's `lm`in Python many years ago. It was called `lmpy` at the time. I wanted to add `lme` and `gam` too, but didn't manage to go very far, mostly because the Python ecosystem back then still lacked general supports for generating the needed design matrices to fit such models. I come back to it from time to time, check community progress, tinker with it a bit myself, and slowly, try to do it with many generations of LLMs as they came along. It's kind of becoming my private benchmark for LLMs, and see at which point they can crack code porting from R to Python without too much hand holding. Until recently, they still couldn't. But since Claude Code with Opus 4.6, things started to change.

BTW, the name `lmpy` is blocked on PyPI due to the name is "too similar to some existing projects", so I asked Claude to brainstorm some other names and one of them is "hea, v., to measure" in Hawaiian, which is likely a [hallucination](https://en.wiktionary.org/wiki/hea#Hawaiian). But I like it a lot, as it means something completely different in Cantonese and fit the vibe of this project nicely. :)

## Install

```bash
pip install hea
```

for sparse matrix speed-up in `lme`, you might want to use `scikit-sparse`, but you need to install some system deps first:

```bash
# mac
brew install suite-sparse

# debian
sudo apt-get install libsuitesparse-dev
```

then 

```bash
pip install "hea[fast]"
```

## Usage

```python
from hea import lm, data

gala = data('gala', package='faraway')
m = lm('Species ~ Area + Adjacent + Elevation + Nearest + Scruz', gala)
m.summary()
```

See more [examples](./example/).