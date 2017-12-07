# encoding:utf-8
import tensorflow as tf


class TensorFlowDKT(object):
    def __init__(self, config):
        self.hidden_neurons = hidden_neurons = config["hidden_neurons"]
        self.num_skills = num_skills = config["num_skills"]
        self.input_size = input_size = config["input_size"]
        self.batch_size = batch_size = config["batch_size"]
        self.keep_prob_value = config["keep_prob"]

        self.max_steps = tf.placeholder(tf.int32)  # max seq length of current batch
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size])
        self.sequence_len = tf.placeholder(tf.int32, [batch_size])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob

        self.target_id = tf.placeholder(tf.int32, [batch_size, None])
        self.target_correctness = tf.placeholder(tf.float32, [batch_size, None])

        # create rnn cell
        hidden_layers = []
        for idx, hidden_size in enumerate(hidden_neurons):
            lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer,
                                                         output_keep_prob=self.keep_prob)
            hidden_layers.append(hidden_layer)
        self.hidden_cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

        # dynamic rnn
        state_series, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                             inputs=self.input_data,
                                                             sequence_length=self.sequence_len,
                                                             dtype=tf.float32)

        # output layer
        output_w = tf.get_variable("W", [hidden_neurons[-1], num_skills])
        output_b = tf.get_variable("b", [num_skills])
        self.state_series = tf.reshape(state_series, [batch_size * self.max_steps, hidden_neurons[-1]])
        self.logits = tf.matmul(self.state_series, output_w) + output_b
        self.mat_logits = tf.reshape(self.logits, [batch_size, self.max_steps, num_skills])
        self.pred_all = tf.sigmoid(self.mat_logits)

        # compute loss
        flat_logits = tf.reshape(self.logits, [-1])
        flat_target_correctness = tf.reshape(self.target_correctness, [-1])
        flat_base_target_index = tf.range(batch_size * self.max_steps) * num_skills
        flat_bias_target_id = tf.reshape(self.target_id, [-1])
        flat_target_id = flat_bias_target_id + flat_base_target_index
        flat_target_logits = tf.gather(flat_logits, flat_target_id)
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [batch_size, self.max_steps]))
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.int32)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness,
                                                                          logits=flat_target_logits))

        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 4)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.grads, trainable_vars))

    # step on batch
    def step(self, sess, input_x, target_id, target_correctness, sequence_len, is_train):
        _, max_steps, _ = input_x.shape
        input_feed = {self.input_data: input_x,
                      self.target_id: target_id,
                      self.target_correctness: target_correctness,
                      self.max_steps: max_steps,
                      self.sequence_len: sequence_len}

        if is_train:
            input_feed[self.keep_prob] = self.keep_prob_value
            train_loss, _, _ = sess.run([self.loss, self.train_op, self.current_state], input_feed)
            return train_loss
        else:
            input_feed[self.keep_prob] = 1
            bin_pred, pred, pred_all = sess.run([self.binary_pred, self.pred, self.pred_all], input_feed)
            return bin_pred, pred, pred_all

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
