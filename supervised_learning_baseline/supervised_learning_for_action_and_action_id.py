import tensorflow as tf
import numpy as np
from batch_generator import batchGenerator


minimap_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 5])
screen_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 10])
user_info_placeholder = tf.placeholder(tf.float32, [None, 11])
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

# user info
l1_user_info = tf.layers.dense(user_info_placeholder, 11, tf.tanh)
user_info_output = tf.layers.dense(l1_user_info, 5)

# regression, NOT SURE IF THIS IS suitable regression
input_to_regression = tf.concat([minimap_output, screen_output, user_info_output], 1)

l2_classification = tf.layers.dense(input_to_classification, 1024, tf.nn.relu)
classification_output = tf.layers.dense(l2_classification, 524)              # output layer
loss = tf.losses.softmax_cross_entropy(onehot_labels=action_output, logits=classification_output)

training_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
tf.summary.scalar('loss', loss) # add loss to scalar summary

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(action_output, axis=1), predictions=tf.argmax(classification_output, axis=1),)[1]


sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph
saver = tf.train.Saver() # define a saver for saving and restoring
writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
merge_op = tf.summary.merge_all() # operation to merge all summary

bg = batchGenerator()
for step in range(1000):                             # train
    m,s,u,a =  bg.next_batch(get_action_id_only=True)
    _, loss_, result = sess.run([train_op, loss, merge_op],
        {minimap_placeholder: m, 
        screen_placeholder: s, 
        user_info_placeholder:u,
        action_output: a})
    writer.add_summary(result, step)

    if step % 50 == 0:
        accuracy_ = sess.run([accuracy],
            {minimap_placeholder: m, 
            screen_placeholder: s, 
            user_info_placeholder:u,
            action_output: a})
    print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

saver.save(sess, './params', write_meta_graph=False)  # meta_graph is not recommended





