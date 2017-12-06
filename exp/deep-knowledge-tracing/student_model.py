# The code is rewritten based on source code from tensorflow tutorial for Recurrent Neural Network.
# https://www.tensorflow.org/versions/0.6.0/tutorials/recurrent/index.html
# You can get source code for the tutorial from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
#
# There is dropout on each hidden layer to prevent the model from overfitting
#
# Here is an useful practical guide for training dropout networks
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# You can find the practical guide on Appendix A
import numpy as np
import tensorflow as tf
import time
import csv
from random import shuffle
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt

# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.3, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.6, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "The number of hidden layers (Integer)")
tf.flags.DEFINE_integer("hidden_size", 200, "The number of hidden nodes (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 150, "Number of epochs to train for.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("train_data_path", 'data/0910_b_train.csv', "Path to the training dataset")
tf.flags.DEFINE_string("test_data_path", 'data/0910_b_test.csv', "Path to the testing dataset")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class StudentModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = FLAGS.batch_size
        self.num_skills = num_skills = config.num_skills
        self.hidden_size = size = FLAGS.hidden_size
        self.num_steps = num_steps = config.num_steps
        input_size = num_skills*2

        inputs = self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._target_id = target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])
        final_hidden_size = size

        hidden_layers = []
        for i in range(FLAGS.hidden_layer_num):
            final_hidden_size = size/(i+1)
            hidden1 = tf.nn.rnn_cell.LSTMCell(final_hidden_size, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                hidden1 = tf.nn.rnn_cell.DropoutWrapper(hidden1, output_keep_prob=FLAGS.keep_prob)
            hidden_layers.append(hidden1)

        cell = tf.nn.rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)

        input_data = tf.reshape(self._input_data, [-1])
        #one-hot encoding
        with tf.device("/cpu:0"):
            labels = tf.expand_dims(input_data, 1)
            indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
            concated = tf.concat(1, [indices, labels])
            inputs = tf.sparse_to_dense(concated, tf.pack([batch_size*num_steps, input_size]), 1.0, 0.0)
            inputs.set_shape([batch_size*num_steps, input_size])

        # [batch_size, num_steps, input_size]
        inputs = tf.reshape(inputs, [-1, num_steps, input_size])
        x = tf.transpose(inputs, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, input_size])
        # Split to get a list of 'n_steps'
        # tensors of shape (doc_num, n_input)
        x = tf.split(0, num_steps, x)
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        #outputs, state = tf.nn.rnn(hidden1, x, dtype=tf.float32)
        outputs, state = tf.nn.rnn(cell, x, dtype=tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, final_hidden_size])
        # calculate the logits from last hidden layer to output layer
        sigmoid_w = tf.get_variable("sigmoid_w", [final_hidden_size, num_skills])
        sigmoid_b = tf.get_variable("sigmoid_b", [num_skills])
        logits = tf.matmul(output, sigmoid_w) + sigmoid_b

        # from output nodes to pick up the right one we want
        logits = tf.reshape(logits, [-1])
        selected_logits = tf.gather(logits, self.target_id)

        #make prediction
        self._pred = self._pred_values = pred_values = tf.sigmoid(selected_logits)

        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(selected_logits, target_correctness))

        #self._cost = cost = tf.reduce_mean(loss)
        self._cost = cost = loss

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_data(self):
        return self._input_data

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def pred_values(self):
        return self._pred_values

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

class HyperParamsConfig(object):
    """Small config."""
    init_scale = 0.05
    num_steps = 0
    max_grad_norm = FLAGS.max_grad_norm
    max_max_epoch = FLAGS.epochs
    keep_prob = FLAGS.keep_prob
    num_skills = 0

def run_epoch(session, m, students, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()

    index = 0
    pred_labels = []
    actual_labels = []
    while(index+m.batch_size < len(students)):
        x = np.zeros((m.batch_size, m.num_steps))
        target_id = []
        target_correctness = []
        count = 0
        for i in range(m.batch_size):
            student = students[index+i]
            problem_ids = student[1]
            correctness = student[2]
            for j in range(len(problem_ids)-1):
                problem_id = int(problem_ids[j])
                label_index = 0
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + m.num_skills
                x[i, j] = label_index
                target_id.append(i*m.num_steps*m.num_skills+j*m.num_skills+int(problem_ids[j+1]))
                target_correctness.append(int(correctness[j+1]))
                actual_labels.append(int(correctness[j+1]))

        index += m.batch_size

        pred, _ = session.run([m.pred, eval_op], feed_dict={
            m.input_data: x, m.target_id: target_id,
            m.target_correctness: target_correctness})

        for p in pred:
            pred_labels.append(p)
    #print pred_labels
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    #calculate r^2
    r2 = r2_score(actual_labels, pred_labels)
    return rmse, auc, r2


def read_data_from_csv_file(fileName):
    config = HyperParamsConfig()
    inputs = []
    targets = []
    rows = []
    max_skill_num = 0
    max_num_problems = 0
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    index = 0
    i = 0
    print "the number of rows is " + str(len(rows))
    tuple_rows = []
    #turn list to tuple
    while(index < len(rows)-1):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index+1]))
        if(tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
        if(problems_num <= 2):
            index += 3
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index], rows[index+1], rows[index+2])
            tuple_rows.append(tup)
            index += 3
    #shuffle the tuple

    random.shuffle(tuple_rows)
    print "The number of students is ", len(tuple_rows)
    print "Finish reading data"
    return tuple_rows, max_num_problems, max_skill_num+1

def main(unused_args):
    config = HyperParamsConfig()
    eval_config = HyperParamsConfig()
    timestamp = str(time.time())
    train_data_path = FLAGS.train_data_path
    #path to your test data set
    test_data_path = FLAGS.test_data_path
    #the file to store your test results
    result_file_path = "run_logs_{}".format(timestamp)
    #your model name
    model_name = "DKT"

    train_students, train_max_num_problems, train_max_skill_num = read_data_from_csv_file(train_data_path)
    config.num_steps = train_max_num_problems
    
    config.num_skills = train_max_skill_num
    test_students, test_max_num_problems, test_max_skill_num = read_data_from_csv_file(test_data_path)
    eval_config.num_steps = test_max_num_problems
    eval_config.num_skills = test_max_skill_num

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # decay learning rate
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

        with tf.Session(config=session_conf) as session:

            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

            # training model
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = StudentModel(is_training=True, config=config)
            # testing model
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = StudentModel(is_training=False, config=eval_config)

            grads_and_vars = optimizer.compute_gradients(m.cost)
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
            session.run(tf.initialize_all_variables())
            # log hyperparameters to results file
            with open(result_file_path, "a+") as f:
                print("Writing hyperparameters into file")
                f.write("Hidden layer size: %d \n" % (FLAGS.hidden_size))
                f.write("Dropout rate: %.3f \n" % (FLAGS.keep_prob))
                f.write("Batch size: %d \n" % (FLAGS.batch_size))
                f.write("Max grad norm: %d \n" % (FLAGS.max_grad_norm))
            saver = tf.train.Saver(tf.all_variables())

            for i in range(config.max_max_epoch):
                rmse, auc, r2 = run_epoch(session, m, train_students, train_op, verbose=True)
                print("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (i + 1, rmse, auc, r2))

                if((i+1) % FLAGS.evaluation_interval == 0):
                    print "Save variables to disk"
                    save_path = saver.save(session, model_name)
                    print("*"*10)
                    print("Start to test model....")
                    rmse, auc, r2 = run_epoch(session, mtest, test_students, tf.no_op())
                    print("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % (i+1, rmse, auc, r2))
                    with open(result_file_path, "a+") as f:
                        f.write("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f" % ((i+1)/2, rmse, auc, r2))
                        f.write("\n")

                        print("*"*10)

if __name__ == "__main__":
    tf.app.run()
