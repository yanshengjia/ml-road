# Deep Knowledge Tracing Implementation
This repository contains our implementation of the [Deep Knowledge Tracing (DKT)](http://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf) model
which we used in [our paper](https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf).

DKT is a recurrent neural network model designed to predict students' performance. 
The authors of the DKT paper used a Long-Term Short-Term (LSTM) network in the paper but 
only [published code](https://github.com/chrispiech/DeepKnowledgeTracing)
for a simple recurrent neural network. In our paper we compare various enhanced
flavors of Bayesian Knowledge Tracing (BKT) to DKT. To ensure a fair comparison to DKT,
we implemented our own LSTM variant of DKT. This repository contains our implementation.


# Requirements
 - Python
 - Theano
 - Keras
 - numpy
 - scikit-learn

# Data Format
The model is contained within one script. It expects two files: a dataset file and
a split file. The dataset file contains student, skill and performance information whilst the split file specifies which students belong to the training set. 

The dataset file is a 3-column space-delimited file. Each row in the file indicates whether a particular student answered a specific problem correctly or not. 
The first column is the student id, the second column is the skill id associated with the problem and the last column is whether the student got the problem correctly (1) or not (0).

The split file is a space-delimited file where each column indicates whether the corresponding student id should be the training set (1) or not (0). For example, the split file:

    1 1 0 1 1 0
    
indicates that students 0, 1, 3 and 4 should be the training set and the rest will be in the test set. 

# Usage

```sh
python dkt.py [-h] --dataset DATASET --splitfile SPLITFILE
              [--hiddenunits HIDDENUNITS] [--batchsize BATCHSIZE]
              [--timewindow TIMEWINDOW] [--epochs EPOCHS]
```

The script will emit three files:
 - **DATASET.model_weights**: contains neural network weights
 - **DATASET.history**: a two column file where the first column contains training log likelihood and the second the test AUC. Each row corresponds to an epoch.
 - **DATASET.preds**: a two column file where the first column contains model prediction and the second the actual observation. Each row corresponds to a test trial.

You can take advantage of the GPU, if you have an nvidia card, by using the command:

```sh
THEANO_FLAGS="device=gpu,floatX=float32" python dkt.py ...
```

# Datasets

We have included the skill builder version of the [Assistments 2009-2010 dataset](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010)
which is one of the datasets evaluated in the paper.