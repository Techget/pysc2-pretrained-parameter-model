import tensorflow as tf
import numpy as np
from batch_generator import batchGenerator
import pysc2.lib.actions as pysc2_actions


minimap_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 5])
screen_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 10])
user_info_placeholder = tf.placeholder(tf.float32, [None, 11])
available_action_placeholder = tf.placeholder(tf.float32, [None, len(pysc2_actions.FUNCTIONS)])

action_output = tf.placeholder(tf.float32, [None, 524]) # one hot
# X_Y_ouput = tf.placeholder([-1, 2])

# minimap
conv1_minimap = tf.layers.conv2d(   
    inputs=minimap_placeholder,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.leaky_relu
)           # -> (64, 64, 16)
pool1_minimap = tf.layers.max_pooling2d(
    conv1_minimap,
    pool_size=2,
    strides=2,
)           # -> (32, 32, 16)
conv2_minimap = tf.layers.conv2d(pool1_minimap, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (32, 32, 32)
pool2_minimap = tf.layers.max_pooling2d(conv2_minimap, 2, 2)    # -> (16, 16, 32)
flat_minimap = tf.reshape(pool2_minimap, [-1, 16*16*32])          # -> (16*14632, )
dense_minimap = tf.layers.dense(inputs=flat_minimap, units=1024, activation=tf.nn.relu)
# dropout_mininmap = tf.layers.dropout(
#     inputs=dense_minimap, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
minimap_output = tf.layers.dense(dense_minimap, 256)

# screen
conv1_screen = tf.layers.conv2d(   
    inputs=screen_placeholder,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.leaky_relu
)           # -> (64, 64, 16)
pool1_screen = tf.layers.max_pooling2d(
    conv1_screen,
    pool_size=2,
    strides=2,
)           # -> (32, 32, 16)
conv2_screen = tf.layers.conv2d(pool1_screen, 32, 5, 1, 'same', activation=tf.nn.relu) # -> (32, 32, 32)
pool2_screen = tf.layers.max_pooling2d(conv2_screen, 2, 2)    # -> (16, 16, 32)
flat_screen = tf.reshape(pool2_screen, [-1, 16*16*32])          # -> (16*16*32, )
dense_screen = tf.layers.dense(inputs=flat_screen, units=1024, activation=tf.nn.relu)
# dropout_screen = tf.layers.dropout(
#     inputs=dense_screen, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
screen_output = tf.layers.dense(dense_screen, 256)

# avaliable actions
l1_available_actions = tf.layers.dense(available_action_placeholder, 128, tf.nn.relu)
avaliable_actions_output = tf.layers.dense(l1_available_actions, 60, tf.nn.relu)

# user info
l1_user_info = tf.layers.dense(user_info_placeholder, 11, tf.tanh)
user_info_output = tf.layers.dense(l1_user_info, 4)


# regression, NOT SURE IF THIS IS suitable regression
input_to_classification = tf.concat([minimap_output, screen_output, avaliable_actions_output, user_info_output], 1)

# LSTM
# RNN_HIDDEN = 256+256+60+4 # 576 = 24 * 24
HIDDEN_SIZE = 512
KEEP_PROB = 0.7
LAYER_NUM = 2
input_to_rnn = tf.reshape(input_to_classification, [-1, 24, 24])
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=KEEP_PROB)
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * LAYER_NUM, state_is_tuple=True)
batch_size = tf.shape(input_to_rnn)[0]
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
# rnn_outputs, rnn_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=input_to_rnn, initial_state=init_state, time_major=False)
inputs_rnn = tf.unstack(input_to_rnn, num=self.num_steps, axis=1)
rnn_outputs, rnn_state = tf.nn.static_rnn(mlstm_cell, inputs_rnn, initial_state=init_state)
input_to_classification = rnn_state[-1][1] # shape is [batch_size, HIDDEN_SIZE]

# lstm_layer = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, forget_bias=1, state_is_tuple=True)
# batch_size    = tf.shape(input_to_classification)[1]
# initial_state = lstm_layer.zero_state(batch_size, tf.float32)
# # outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
# rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm_layer, input_to_classification, initial_state=initial_state, time_major=True)
# input_to_classification = rnn_outputs

l2_classification = tf.layers.dense(input_to_classification, 1024, tf.nn.relu)
classification_output = tf.layers.dense(l2_classification, 524)              # output layer
loss = tf.losses.softmax_cross_entropy(onehot_labels=action_output, logits=classification_output)

# want to maximize this valid_action_loss
valid_action_loss = tf.reduce_sum(tf.reduce_sum(classification_output * available_action_placeholder, axis=1))

# multiply by -1 so that it fit for minimize
loss += valid_action_loss * (-1)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
tf.summary.scalar('loss', loss) # add loss to scalar summary

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(action_output, axis=1), predictions=tf.argmax(classification_output, axis=1),)[1]


sess = tf.Session()                                 # control training and others
# sess.run(tf.global_variables_initializer(), tf.local_variables_initializer())    # initialize var in graph
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op) # initialize var in graph

saver = tf.train.Saver() # define a saver for saving and restoring
writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
merge_op = tf.summary.merge_all() # operation to merge all summary

bg = batchGenerator()
for step in range(5000):                             # train
    m,s,u,aa,a =  bg.next_batch()
    _, loss_, result = sess.run([train_op, loss, merge_op],
        {minimap_placeholder: m, 
        screen_placeholder: s, 
        user_info_placeholder:u,
        available_action_placeholder:aa,
        action_output: a})

    if step % 50 == 0:
        m,s,u,aa,a =  bg.next_batch(get_validation_data=True)
        accuracy_ = sess.run([accuracy],
            {minimap_placeholder: m, 
            screen_placeholder: s, 
            user_info_placeholder:u,
            available_action_placeholder:aa,
            action_output: a})
        print('Step:', step,'| loss_: ', loss_, '| test accuracy: ', accuracy_)

    writer.add_summary(result, step)
    # print('~~~~~')

saver.save(sess, './action_id')  # meta_graph is not recommended, , write_meta_graph=False





