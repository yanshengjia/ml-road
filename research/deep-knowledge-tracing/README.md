# Deep Knowledge Tracing

A tensorflow implementation of the LSTM version of Deep Knowledge Tracing (DKT).

Also, the LSTM model is optimized by BPTT (Back Propagation Through Time).

## Requirements

* tensorflow >= 1.0.0
* scikit-learn >= 0.17.1

## Uasge

```shell
python train_dkt.py --dataset ../data/assistments.txt
```

## References

* [Deep Knowledge Tracing (DKT)](http://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf)
* [lingochamp/tensorflow-dkt](https://github.com/lingochamp/tensorflow-dkt)
* [Backpropagation-through-time library for TensorFlow](https://github.com/devnag/tensorflow-bptt)

