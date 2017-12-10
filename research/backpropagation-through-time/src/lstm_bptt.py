#!/usr/bin/env python

import math
import numpy as np
import random
import sys
import tensorflow as tf
import bptt

# See https://medium.com/@devnag/


# Data parameters: simple one-number-at-a-time for now
input_dimensions = 1
output_dimensions = 1
batch_size = 1

# Model parameters
lstm_width = 5
m = 0.0
s = 0.5
init = tf.random_normal_initializer(m, s)
noise_m = 0.0
noise_s = 0.03

# Optimization parameters
learning_rate = 0.05
beta1 = 0.95
beta2 = .999
epsilon = 1e-3
momentum = 0.4
gradient_clipping = 2.0
unroll_depth = 4
max_reset_loops = 20

# Training parameters
num_training_loops = 3000
num_inference_loops = 100
num_inference_warmup_loops = 1900


def build_lstm_layer(bp, depth_type, layer_index, raw_x, width):
    """
    Build a single LSTM layer (Graves 2013); can be stacked, but send in sequential layer_indexes to scope properly.
    """
    global init, noise_m, noise_s
    # Define variable names
    h_name = "hidden-%s" % layer_index  # Really the 'output' of the LSTM layer
    c_name = "cell-%s" % layer_index
    # raw_x is [input_size, 1]
    input_size = raw_x.get_shape()[0].value
    # Why so serious? Introduce a little anarchy. Upset the established order...
    x = raw_x + tf.random_normal(raw_x.get_shape(), noise_m, noise_s)

    with tf.variable_scope("lstm_layer_%s" % layer_index):

        # Define shapes for all the weights/biases, limited to just this layer (not shared with other layers)
        # Sizes are 'input_size' when mapping x and 'width' otherwise
        W_xi = tf.get_variable("W_xi", [width, input_size], initializer=init)
        W_hi = tf.get_variable("W_hi", [width, width], initializer=init)
        W_ci = tf.get_variable("W_ci", [width, width], initializer=init)
        b_i =  tf.get_variable("b_i",  [width, 1], initializer=init)
        W_xf = tf.get_variable("W_xf", [width, input_size], initializer=init)
        W_hf = tf.get_variable("W_hf", [width, width], initializer=init)
        W_cf = tf.get_variable("W_cf", [width, width], initializer=init)
        b_f =  tf.get_variable("b_f",  [width, 1], initializer=init)
        W_xc = tf.get_variable("W_xc", [width, input_size], initializer=init)
        W_hc = tf.get_variable("W_hc", [width, width], initializer=init)
        b_c =  tf.get_variable("b_c",  [width, 1], initializer=init)
        W_xo = tf.get_variable("W_xo", [width, input_size], initializer=init)
        W_ho = tf.get_variable("W_ho", [width, width], initializer=init)
        W_co = tf.get_variable("W_co", [width, width], initializer=init)
        b_o =  tf.get_variable("b_o",  [width, 1], initializer=init)

        # Retrieve the previous roll-depth's data, with starting random data if first roll-depth.
        h_past = bp.get_past_variable(h_name, np.float32(np.random.normal(m, s, [width, 1])))
        c_past = bp.get_past_variable(c_name, np.float32(np.random.normal(m, s, [width, 1])))

        # Build graph - looks almost like Alex Graves wrote it!
        i = tf.sigmoid(tf.matmul(W_xi, x) + tf.matmul(W_hi, h_past) + tf.matmul(W_ci, c_past) + b_i)
        f = tf.sigmoid(tf.matmul(W_xf, x) + tf.matmul(W_hf, h_past) + tf.matmul(W_cf, c_past) + b_f)
        c = bp.name_variable(c_name, tf.multiply(f, c_past) + tf.multiply(i, tf.tanh(tf.matmul(W_xc, x) + tf.matmul(W_hc, h_past) + b_c)))
        o = tf.sigmoid(tf.matmul(W_xo, x) + tf.matmul(W_ho, h_past) + tf.matmul(W_co, c) + b_o)
        h = bp.name_variable(h_name, tf.multiply(o, tf.tanh(c)))

    return [c, h]


def build_dual_lstm_frame(bp, depth_type):
    """
    Build a dual-layer LSTM followed by standard sigmoid/linear mapping
    """
    global init, input_dimensions, output_dimensions, batch_size, lstm_width

    # I/O DATA
    input_placeholder = tf.placeholder(tf.float32, shape=(input_dimensions, batch_size))
    output_placeholder = tf.placeholder(tf.float32, shape=(output_dimensions, batch_size))

    last_output = input_placeholder
    for layer_index in xrange(2):
        [_, h] = build_lstm_layer(bp, depth_type, layer_index, last_output, lstm_width)
        last_output = h

    W = tf.get_variable("W", [1, lstm_width], initializer=init)
    b = tf.get_variable("b", [1,1], initializer=init)
    output_result = tf.sigmoid(tf.matmul(W, last_output) + b)

    # return array of whatever you want, but I/O placeholders FIRST.
    return [input_placeholder, output_placeholder, output_result]


def palindrome(step):
    """
    Turn sequential integers into a palindromic sequence (so look-ahead mapping is not a function, but requires state)
    """
    return (5.0 - abs(float(step % 10) - 5.0)) / 10.0


bp = None
sess = None
graphs = None
done = False

# Loop until you get out of a local minimum or you hit max reset loops
for reset_loop_index in xrange(max_reset_loops):

    # Clean any previous loops
    if reset_loop_index > 0:
        tf.reset_default_graph()

    # Generate unrolled+shallow graphs
    bp = bptt.BPTT()
    graphs = bp.generate_graphs(build_dual_lstm_frame, unroll_depth)

    # Define loss and clip gradients
    error_vec = [[o - p] for [i, p, o] in graphs[bp.DEEP]]
    loss = tf.reduce_mean(tf.square(error_vec))
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
    grads = optimizer.compute_gradients(loss)
    clipped_grads = [(tf.clip_by_value(grad, -gradient_clipping, gradient_clipping), var) for grad, var in grads]
    optimizer.apply_gradients(clipped_grads)
    train = optimizer.minimize(loss)

    # Boilerplate initialization
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    reset = False

    print("=== Training the unrolled model (reset loop %s) ===" % (reset_loop_index))

    for step in xrange(num_training_loops):
        # 1.) Generate the dictionary of I/O placeholder data
        start_index = step * unroll_depth
        in_data = np.array([palindrome(x) for x in xrange(start_index, start_index + unroll_depth)], dtype=np.float32)
        out_data = np.array([palindrome(x+1) for x in xrange(start_index, start_index + unroll_depth)], dtype=np.float32)

        # 2a.) Generate the working state to send in, along with data to insert into unrolled placeholders
        frame_dict = bp.generate_feed_dict(bp.DEEP, [in_data, out_data], 2)

        # 2b.) Define the output (training/loss) that we'd like to see (optional)
        session_out = [train, loss] + [o for [i, p, o] in graphs[bp.DEEP]]   # calculated output

        # 3.) Define state variables to pull out as well.
        state_vars = bp.generate_output_definitions(bp.DEEP)
        session_out.extend(state_vars)

        # 4.) Execute the graph
        results = sess.run(session_out, feed_dict=frame_dict)

        # 5.) Extract the state for next training loop; need to make sure we have right part of result array
        bp.save_output_state(bp.DEEP, results[-len(state_vars):])  # for simple RNN

        # 6.) Show training progress; reset graph if loss is stagnant.
        if (step % 100) == 0:
            print "Loss: %s => %s (output: %s)" % (step, results[1], [str(x) for x in results[2:-len(state_vars)]])
            sys.stdout.flush()

            if step >= 1000 and (results[1] > 0.01):
                print("\nResetting; loss (%s) is stagnating after 1k rounds...\n" % (results[1]))
                reset = True
                break  # To next reset loop

    if not reset:
        break

print("=== Evaluating on shallow model ===")

# Copy final deep state from the training loop above to the shallow state.
bp.copy_state_forward()
[in_ph, out_ph, out_out] = graphs[bp.SHALLOW][0]

# Evaluate one step at a time, and burn in first.
for step in xrange(num_inference_loops + num_inference_warmup_loops):
    # 1.) Convert step to the palindromic sequence (current and look-ahead-by-one)
    in_value = palindrome(step)
    expected_out_value = palindrome(step+1)

    # 2.) Generate the feed dictionary to send in, both I/O data and recurrent variables
    frame_dict = bp.generate_feed_dict(bp.SHALLOW, np.array([[in_value]], np.float32), 1)

    # 3.) Define state variables to pull out
    session_out = [out_out]
    state_vars = bp.generate_output_definitions(bp.SHALLOW)
    session_out.extend(state_vars)

    # 4.) Execute the graph
    results = sess.run(session_out, feed_dict=frame_dict)

    # 5.) Extract/save state variables for the next loop
    bp.save_output_state(bp.SHALLOW, results[-len(state_vars):])

    # 6.) How we doin'?
    if step > num_inference_warmup_loops:
        print("%s: %s => %s actual vs %s expected (diff: %s)" %
              (step, in_value, results[0][0][0], expected_out_value, expected_out_value - results[0][0][0]))
        sys.stdout.flush()




