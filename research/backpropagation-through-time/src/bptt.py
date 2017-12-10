#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# See https://medium.com/@devnag/


class BPTT(object):
    """
    Convenience design pattern for handling simple recurrent graphs, implementing backpropagation through time.
    See https://medium.com/@devnag/

    Typical usage:

    - Graph building
        - Define a function that takes a BPTT object and the depth flag (will be BPTT.DEEP or BPTT.SHALLOW)
              and builds your computational graph; should return any I/O placeholders in an array.
          - Use get_past_variable() to define a name (string) and pass in a constant value (numpy).
          - Use name_variable() to name (string) the same value for the current loop, for the future.

    - Unrolling
        - bp.generate_graphs() will take the function above and the desired BPTT depth and provide the
            sequence of stitched DAGs.


    - Training
        - generate_feed_dict() on the relevant depth (BPTT.DEEP) with the array data to be fed into the
           I/O placeholders that your custom graph function returned. This will also include the working
           state for the recurrent variables (whether the starting constants or state from the last loop).
           Must also include a count of the number of I/O slots.
        - generate_output_definitions() will provide an array of variables that must be fetched to extract state.
        - save_output_state() will take the results and save for the next loop.

    - Inference
        - Same three functions as in training, but use BPTT.SHALLOW instead.
        - Can optionally call copy_state_forward() before inference if you want to start with the final training state.
    """

    DEEP = "deep"
    SHALLOW = "shallow"
    MODEL_NAME = "unrolled_model"
    LOOP_SCOPE = "unroll"

    def __init__(self):
        """
        Initialize the name dictionaries (state, placeholders, constants, etc)
        """
        self.graph_dict = {}

        # Name -> Constants: Starting values (typically np.arrays). Shared between shallow/deep, used in run-time
        self.starting_constants = {}
        # Name -> State: np.arrays reflecting state between run-times (starting from C)
        self.state = {self.DEEP: {}, self.SHALLOW: {}}
        # Name -> Variables: Py variables passed through during build-time
        self.vars = {self.DEEP: {}, self.SHALLOW: {}}
        # Name -> Placeholder: Placeholders: to inject state, set during build-time
        self.placeholders = {self.DEEP: {}, self.SHALLOW: {}}

        self.current_depth = self.DEEP

    def get_past_variable(self, variable_name, starting_value):
        """
        Get-or-set a recurrent variable from the past (time t-1)

        :param variable_name: A unique (to this object) string representing this variable.
        :param starting_value: A constant that can be fed into a placeholder eventually
        :return: A variable (representing the value at t-1) that can be computed on to generate current value (at t)
        """

        if variable_name not in self.placeholders[self.current_depth]:
            # First time being called
            self.starting_constants[variable_name] = starting_value

            # First initial state is the constant np.array sent in
            self.state[self.current_depth][variable_name] = starting_value

            # Define a mirror placeholder with same type/shape
            self.placeholders[self.current_depth][variable_name] = tf.placeholder(starting_value.dtype,
                                                                                  shape=starting_value.shape)
            # Set current (starting) variable as that placeholder, to be filled in later
            self.vars[self.current_depth][variable_name] = self.placeholders[self.current_depth][variable_name]

        # Return the pyvariable: placeholder the first time, pydescendant on later calls
        return self.vars[self.current_depth][variable_name]

    def name_variable(self, variable_name, v):
        """
        Set/assign a recurrent variable for the current time (time t)

        :param variable_name: A unique (to this object) string, must have been used in a get_past_variable() call
        :param v: A Tensorflow variable representing the current value of this variable (at t)
        :return: v, unchanged, for easy in-line usage
        """
        assert variable_name in self.vars[self.current_depth], \
            "Tried to set variable name that was never defined with get_past_variable()"
        self.vars[self.current_depth][variable_name] = v
        return v

    def generate_graphs(self, func, num_loops=10):
        """
        Generate the two graphs -- the deep (unrolled) connected graphs and the shallow/simple graph.

        :param func: A function which takes the BPTT object and the depth_type (BPTT.{DEEP,SHALLOW}), returns
                    array of I/O placeholders.
        :param num_loops: The desired number of loops to unroll
        :return: A dictionary of the two graphs (deep+shallow).
        """
        # Scoping -- generate the deep/unrolled graph (training)
        self.current_depth = self.DEEP
        with tf.variable_scope(self.MODEL_NAME, reuse=False):
            self.graph_dict[self.DEEP] = self.unroll(func, self.DEEP, num_loops)

        # Now, generate the shallow graph (inference)
        self.current_depth = self.SHALLOW
        with tf.variable_scope(self.MODEL_NAME, reuse=True):
            # Shallow is depth 1, but sharing all variables with deep graph above
            self.graph_dict[self.SHALLOW] = self.unroll(func, self.SHALLOW, 1)

        return self.graph_dict

    def unroll(self, func, depth_type, num_loops):
        """
        Given the graph-generating function, unroll to the desired depth.

        :param func: A function which takes the BPTT object and the depth_type (BPTT.{DEEP,SHALLOW}), returns
                    array of I/O placeholders.
        :param depth_type: The depth_type (BPTT.{DEEP,SHALLOW})
        :param num_loops: The desired number of loops to unroll
        :return: A list of the graphs, connected by variables.
        """
        frames = []
        for loop in xrange(num_loops):
            # Scoping on top of each depth
            # We need 'False' for the first time and 'True' for all others
            with tf.variable_scope(self.LOOP_SCOPE, reuse=(loop != 0)):
                frames.append(func(self, depth_type))

        return frames

    def generate_feed_dict(self, depth_type, data_array, num_settable):
        """
        Generate a feed dictionary; takes in an array of the data that will be inserted into the unrolled
        placeholders.

        :param depth_type: The depth_type (BPTT.{DEEP,SHALLOW})
        :param data_array: An array of arrays of data to insert into the unrolled placeholders
        :param num_settable: How many elements of the data_array to use.
        :return: A dictionary to feed into tf.Session().run()
        """
        frames = self.graph_dict[depth_type]
        d = {}

        # Recurrent: Auto-defined placeholders / current variables
        for variable_name in self.placeholders[depth_type]:
            d[self.placeholders[depth_type][variable_name]] = self.state[depth_type][variable_name]

        # User-provided data to unroll/insert into the placeholders
        for frame_index in xrange(len(frames)):       # Unroll index
            for var_index in xrange(num_settable):    # Variable index
                frame_var = frames[frame_index][var_index]
                d[frame_var] = np.reshape(data_array[var_index][frame_index],
                                          frame_var.get_shape())
        return d

    def copy_state_forward(self):
        """
        Copy the working state from the DEEP pipeline to the SHALLOW pipeline
        """
        for key in self.state[self.DEEP]:
            self.state[self.SHALLOW][key] = np.copy(self.state[self.DEEP][key])

    def generate_output_definitions(self, depth_type):
        """
        Generate the desired output variables to fetch from the graph run

        :param depth_type: The depth_type (BPTT.{DEEP,SHALLOW})
        :return: An array of variables to add to the fetch list
        """
        d = self.vars[depth_type]
        # Define consistent sort order by the variable names
        return [d[k] for k in sorted(d.keys())]

    def save_output_state(self, depth_type, arr):
        """
        Save the working state for the next run (will be available in generate_feed_dict() in the next loop)

        :param depth_type: The depth_type (BPTT.{DEEP,SHALLOW})
        :param arr: An array of values (returned by tf.Session.run()) which map to generate_output_definitions()
        """
        d = self.state[depth_type]
        sorted_names = sorted(d.keys())
        assert len(sorted_names) == len(arr), \
            "Sent in the wrong number of variables (%s) to update state (%s)" % (len(arr), len(sorted_names))
        for variable_index in xrange(len(sorted_names)):
            variable_name = sorted_names[variable_index]
            # Saved for next time.
            self.state[depth_type][variable_name] = arr[variable_index]


