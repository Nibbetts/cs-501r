import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell, RNNCell
from tensorflow.contrib.legacy_seq2seq import rnn_decoder, sequence_loss


#
# -------------------------------------------
#
# Global variables

batch_size = 50 ###
sequence_length = 50 ###

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2 ###

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, axis=1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

# create a BasicLSTMCell
#   use it to create a MultiRNNCell
#   use it to create an initial_state
#     note that initial_state will be a *list* of tensors!

class mygru( RNNCell ):

    def __init__( self, num_units ):
    	self._num_units = num_units

    @property
    def state_size(self):
    	return self._num_units

    @property
    def output_size(self):
    	return self._num_units

    def __call__( self, inputs, state, scope=None ):
    	# x_t = inputs
        # h_t-1 = state
        batch_size, input_size = inputs.get_shape().as_list()

        W_xr = tf.get_variable(name="W_xr", shape=[input_size, self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())
        W_xz = tf.get_variable(name="W_xz", shape=[input_size, self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())
        W_xh = tf.get_variable(name="W_xh", shape=[input_size, self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())

        W_hr = tf.get_variable(name="W_hr", shape=[self._num_units, self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())
        W_hz = tf.get_variable(name="W_hz", shape=[self._num_units, self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())
        W_hh = tf.get_variable(name="W_hh", shape=[self._num_units, self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())

        b_r = tf.get_variable(name="b_r", shape=[self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())
        b_z = tf.get_variable(name="b_z", shape=[self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())
        b_h = tf.get_variable(name="b_h", shape=[self._num_units], initializer=tf.contrib.layers.variance_scaling_initializer())

        r_t = tf.sign(tf.matmul(inputs,W_xr) + tf.matmul(state,W_hr) + b_r)
        z_t = tf.sign(tf.matmul(inputs,W_xz) + tf.matmul(state,W_hz) + b_z)
        h_hat_t = tf.tanh(tf.matmul(inputs,W_xh) + tf.matmul(r_t * state, W_hh) + b_h)

        h_t = z_t * state + (1-z_t) * h_hat_t

        return h_t, h_t


multi_cell = MultiRNNCell([BasicLSTMCell(state_dim) for i in range(num_layers)])
#multi_cell = MultiRNNCell([mygru(state_dim) for i in range(num_layers)])
initial_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

# call seq2seq.rnn_decoder
outputs, final_state = rnn_decoder(inputs, initial_state, multi_cell)

# transform the list of state outputs to a list of logits.
# use a linear transformation.
weights = tf.get_variable(name="W", shape=[state_dim, vocab_size],
    initializer=tf.contrib.layers.variance_scaling_initializer())
bias = tf.get_variable(name="b", shape=[vocab_size],
    initializer=tf.contrib.layers.variance_scaling_initializer())

logits = [tf.matmul(o, weights) + bias for o in outputs]

# call seq2seq.sequence_loss
loss = sequence_loss(logits, targets, [1.0] * sequence_length)

# create a training op using the Adam optimizer
optim = tf.train.AdamOptimizer(name='adam').minimize(loss)

# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

s_inputs = tf.placeholder( tf.int32, [ 1 ], name="s_inputs")

s_initial_state = multi_cell.zero_state(1.0, dtype=tf.float32)
s_one_hot = tf.one_hot( s_inputs, vocab_size, name="s_input_onehot" )

s_outputs, s_final_state = rnn_decoder([s_one_hot], s_initial_state, multi_cell)
s_probs = [tf.matmul(o, weights) + bias for o in s_outputs]

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        sample = np.argmax( s_probsv[0] ) ###
        #sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):

        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

    print sample( num=60, prime="And " )
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" ) ###
#    print sample( num=60, prime="abcdab" )
#    print sample(num=60, prime="def ")

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
