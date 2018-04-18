import tensorflow as tf
import numpy as np
from batch_generator import batchGenerator


minimap_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 5])
screen_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 10])
user_info_placeholder = tf.placeholder(tf.float32, [None, 11])
action_placeholder = tf.placeholder(tf.float32, [None, 524]) # one hot

# 
arg_screen_replay_ouput = tf.placeholder(tf.float32, [None, 2])
# arg_screen2_replay_ouput = tf.placeholder(tf.float32, [None, 2])
# arg_minimap_replay_ouput = tf.placeholder(tf.float32, [None, 2])
# arg_queued_replay_output = tf.placeholder(tf.float32, [None, 1])
# arg_control_group_act_replay_output = tf.placeholder(tf.float32, [None, 1])

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
conv2_minimap = tf.layers.conv2d(pool1_minimap, 32, 5, 1, 'same', activation=tf.nn.leaky_relu)    # -> (32, 32, 32)
pool2_minimap = tf.layers.max_pooling2d(conv2_minimap, 2, 2)    # -> (16, 16, 32)
flat_minimap = tf.reshape(pool2_minimap, [-1, 16*16*32])          # -> (16*14632, )
dense_minimap = tf.layers.dense(inputs=flat_minimap, units=1024, activation=tf.nn.leaky_relu)
# dropout_mininmap = tf.layers.dropout(
#     inputs=dense_minimap, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
minimap_output = tf.layers.dense(dense_minimap, 64)

# screen
conv1_screen = tf.layers.conv2d(   
    inputs=screen_placeholder,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
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
screen_output = tf.layers.dense(dense_screen, 64)

# action id
l1_action = tf.layers.dense(action_placeholder, 128, tf.nn.relu)          # hidden layer
l2_action = tf.layers.dense(l1_action, 64, tf.nn.relu)
action_output = tf.layers.dense(l2_action, 10) # output layer

# user info
l1_user_info = tf.layers.dense(user_info_placeholder, 11, tf.tanh)
user_info_output = tf.layers.dense(l1_user_info, 5)

# processed concatenated input
concat_input = tf.concat([minimap_output, screen_output, action_output, user_info_output], 1)

##### arg Types
# screen 
# arg_screen_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
# arg_screen_output = tf.layers.dense(arg_screen_dense, 2)
# arg_screen_loss = tf.reduce_mean(tf.square(arg_screen_output - X_Y_ouput))
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(arg_screen_loss)
# tf.summary.scalar('loss', loss) # add loss to scalar summary

# minimap 

# screen2

# queued

# control_group_act

# control_group_id

# select_point_act

# select_add

# select_unit_act

# select_unit_id

# select_worker

# build_queue_id

# unload_id

regression_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
# dropout_regression = tf.layers.dropout(
#     inputs=dense_screen, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
regression_output = tf.layers.dense(regression_dense, 2)

# loss
loss = tf.reduce_mean(tf.square(regression_output - X_Y_ouput))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
tf.summary.scalar('loss', loss) # add loss to scalar summary

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph
saver = tf.train.Saver() # define a saver for saving and restoring
writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
merge_op = tf.summary.merge_all() # operation to merge all summary


bg = batchGenerator()
for step in range(1000):                             # train
    m,s,a,u,y =  bg.next_batch()
    _, loss_, result = sess.run([train_op, loss, merge_op],
        {minimap_placeholder: m, 
        screen_placeholder: s, 
        action_placeholder: a, 
        user_info_placeholder:u, 
        X_Y_ouput:y})
    writer.add_summary(result, step)
    print('step: ', step, 'loss: ',loss_, 'result: ', result)

saver.save(sess, './params', write_meta_graph=False)  # meta_graph is not recommended





