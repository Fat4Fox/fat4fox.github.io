---
layout: post
title: Axis in Numpy
date:   2021-10-14 11:33:53
categories:
- Numpy
- Axis
---
Many operations in Numpy involves axis, e.g. sum, concatenate. 

In these operations, the axis determines the direction of collapse 
(for aggregation) or expansion (for stacking). Below is an illustration.

![Sum example along axis](/resources/images/axis.jpeg)

PyTorch operations use `dim` argument. It works similar to `axis`.
```python
In [1]: a = torch.rand(3,3)
In [2]: a
Out[2]:
tensor([[0.6295, 0.0995, 0.9350],
        [0.7498, 0.7338, 0.2076],
        [0.2302, 0.7524, 0.1993]])
In [3]: a.shape
Out[3]: torch.Size([3, 3])
In [4]: torch.argmax(a, dim=0)
Out[4]: tensor([1, 2, 0])
In [5]: torch.argmax(a, dim=1)
Out[5]: tensor([2, 0, 1])
In [6]: torch.argmin(a, dim=0)
Out[6]: tensor([2, 0, 2])
In [7]: torch.argmin(a, dim=1)
Out[7]: tensor([1, 2, 2])
```
