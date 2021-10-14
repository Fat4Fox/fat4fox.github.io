---
layout: post
title:  "Tricky PyTorch"
date:   2021-10-13 10:57:28
categories: 
- PyTorch
---

### Your Model Class Inherit from Module
```python
class MLP(Module):

    def __init__(self, n_inputs):
        super(MLP, self).__init__() # backward compatible with python 2.x 
```

### Your Custom Dataset Class
#### Label vector may need reshape
```python
    self.y = LabelEncoder().fit_transform(self.y)
    self.y = self.y.astype('float32')
    self.y = self.y.reshape((len(self.y), 1))
```
