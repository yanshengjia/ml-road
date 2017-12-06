# [Going Deeper with Deep Knowledge Tracing](http://www.educationaldatamining.org/EDM2016/proceedings/paper_133.pdf) - EDM-2016
Original Github [repo](https://github.com/siyuanzhao/2016-EDM).

Source code and data sets for [Going Deeper with Deep Knowledge Tracing](http://www.educationaldatamining.org/EDM2016/proceedings/paper_133.pdf)

### Dependencies:
- Tensorflow 0.10 (only tested on this version)
- Numpy
- scikit-learn

### Get Started

```
git clone https://github.com/siyuanzhao/2016-EDM.git
cd 2016-EDM
python student_model.py
```

### Usage
There are serval flags within student_model.py. Some of them are hyper-parameters for the model. Some of them are path to training and testing data.

Check all available flags with the following command.

```
python student_model.py -h
```

Run the model on a different data set

```
python student_model.py --train_data_path=<path-to-your-data> --test_data_path=<path-to-your-data>
```
You can also set the number of hidden layers and the number of hidden nodes with flags.

```
python student_model.py --hidden_layer_num=2 --hidden_size=400
```

### Details
- The model uses [Adam Optimizer](https://arxiv.org/abs/1412.6980).
- Add gradient noise. [arxiv](http://arxiv.org/abs/1511.06807)
- Add gradient norm clipping. [arxiv](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)

### Datasets
Data sets we used from the paper are in folder data.

<table>
<tr>
  <th></th>
  <th>ASSISTments 09-10 (a)</th>
  <th>ASSISTments 09-10 (b)</th>
  <th>ASSISTments 09-10 (c)</th>
</tr>
<tr>
  <td>file name</td>
  <td>0910_a</td>
  <td>0910_b</td>
  <td>0910_c</td>
</tr>
<tr>
  <td>Has duplicated records</td>
  <td>No</td>
  <td>No</td>
  <td>No</td>
</tr>
<tr>
  <td>Has subproblems</td>
  <td>Yes</td>
  <td>No</td>
  <td>No</td>
</tr>
<tr>
  <td>Repeat sequnces for mutiple skills</td>
  <td>Yes</td>
  <td>Yes</td>
  <td>No</td>
</tr>
<tr>
  <td>Combined skills for mutiple skills</td>
  <td>No</td>
  <td>No</td>
  <td>Yes</td>
</tr>
</table>

CAT_train.csv and CAT_test.csv are data files from Cognitive Tutor.

### Results
Since I made some changes on the code, I will run the model again and record the results.
