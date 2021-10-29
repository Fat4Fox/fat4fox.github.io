---
layout: post
title: Torchtext 0.10 Build Vocab
date:   2021-10-15 16:59:07
categories:
- PyTorch
- Torchtext
---

Torchtext 0.10 release has some major changes to Vocab APIs.

> In this release, we introduce a new Vocab module that replaces the current 
> Vocab class. The new Vocab provides common functional APIs for NLP workflows. 
> This module is backed by an efficient C++ implementation that reduces look-up 
> time by up-to ~85% for batch look-up (refer to summary of #1248 and #1290 for 
> further information on benchmarks), and provides support for TorchScript. 
> We provide accompanying factory functions that can be used to build the 
> Vocab object either through a python ordered dictionary or an Iterator 
> that yields lists of tokens.

The release note also provides two examples for building vocabs.

#### Creating Vocab from text file
```python
import io
from torchtext.vocab import build_vocab_from_iterator
# generator that yield list of tokens
def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
       for line in f:
           yield line.strip().split()
# get Vocab object
vocab_obj = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"])
```

#### Creating Vocab through ordered dict
```python
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
counter = Counter(["a", "a", "b", "b", "b"])
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab_obj = vocab(ordered_dict)
```

### Migration 
Given this, I tried to migrate some old vacab creation code to this new API. 
The old code is from  

Old code
```python
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

tokenizer = get_tokenizer('spacy') # <1>
counter = Counter()
for (label, line) in train_data:
    counter.update(generate_bigrams(
        tokenizer(line))) # <2>
vocab = Vocab(counter, 
              max_size = 25000, 
              vectors = "glove.6B.100d", 
              unk_init = torch.Tensor.normal_) # <3>
```
