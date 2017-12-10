# A Simple Design Pattern for TensorFlow Recurrency

## Introduction

This repo includes the bptt.py library for recurrent graph bookkeeping (in Tensorflow) and a sample client which learns to predict a simple palindromic sequence using a double-layer LSTM (Graves 2013). 

See [this](https://medium.com/@devnag/a-simple-design-pattern-for-recurrent-deep-learning-in-tensorflow-37aba4e2fd6b) blog post for more info.

## Running


```
python lstm_bptt.py
```

you'll train a 2-layer LSTM on a palindromic sequence prediction task, then test it on sequential inference. The loss should drop below 1e-3 pretty quickly, and then you'll see the last few hundred attempts vs. the expected output.

## References

* [devnag/tensorflow-bptt](https://github.com/devnag/tensorflow-bptt)
* https://en.wikipedia.org/wiki/Backpropagation_through_time