---
layout: post
title:  "Tricky PyTorch"
date:   2021-10-13 10:57:28
categories: 
- PyTorch
---
Recently I started to delve deep into PyTorch and found there are many 
tricks. So I log them down here.

### Backbone of a PyTorch Project
- Custom Dataset (if necessary)
```python
    def __init__(self, path): # load data from some path
    def __len__(self): # return number of samples in the entire dataset
    def __getitem__(self, idx): # get a sample at an index
    def get_splits(self, n_test=0.33): # split into train/test datasets
```
- DataLoader (wrapping a Dataset)
```python
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
```
- Model class
```python
    def __init__(self, n_inputs): # defines layers, activations, weight initialization, etc.
    def forward(self, X): # let X pass through the defined layers, activations
```
- Model training
  - loss/criterion, e.g. BCELoss (Binary Cross Entropy Loss)
  - optimizer, e.g. SGD, Adam
```python
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
```
- Model evaluation
```python
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
```
- Inference
```python
    # convert data to a tensor
    row = Tensor([row])
    # pass through the model
    yhat = model(row)
    # retrieve numpy array from the output tensor
    yhat = yhat.detach().numpy()
```

### Your Model Class Inherit from Module
```python
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__() # backward compatible with python 2.x 
```

### Your Custom Dataset Class
```python
    # For Binary classification, reshape to (y, 1)
    self.y = LabelEncoder().fit_transform(self.y)
    self.y = self.y.astype('float32')
    self.y = self.y.reshape((len(self.y), 1))

    # For MultiClass, don't reshape and leave it as (y,), 
    # because PyTorch would do one-hot encoding for the yhat, and later
    # the criterion(yhat, targets) would expect the dimensions like below:
    # yhat.shape: torch.Size([32, 3])
    # targets.shape: torch.Size([32])
    self.y = LabelEncoder().fit_transform(self.y)
```

## Reference
1. [Binary classification sample code](https://colab.research.google.com/drive/1JP6peTrDmVeQB5Sb69UwzkguVywapSuc?usp=sharing)
2. [MultiClass sample code](https://colab.research.google.com/drive/1fB6fGjtA9Gl0bajOUXfuCu6QBfbVA2xA?usp=sharing)
