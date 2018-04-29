import tensorflow as tf
import numpy as np
from batch_generator import batchGenerator
import pysc2.lib.actions as pysc2_actions

LR = 0.0001

minimap_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 5])
screen_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 10])
user_info_placeholder = tf.placeholder(tf.float32, [None, 11])
action_placeholder = tf.placeholder(tf.float32, [None, 524]) # one hot

# arg_placeholder will look like
# [
#     [[0],[1, 2]],
#     [[1],[5]], ...
# ]
# arg_placeholder = tf.placeholder(tf.float32, [None, None, None]) # then assign values in this placeholder to following placeholders

# arg outputs placeholders
arg_screen_replay_ouput = tf.placeholder(tf.float32, [None, 2])
arg_screen2_replay_ouput = tf.placeholder(tf.float32, [None, 2])
arg_minimap_replay_ouput = tf.placeholder(tf.float32, [None, 2])
arg_queued_replay_output = tf.placeholder(tf.int32, [None, 1])
arg_control_group_act_replay_output = tf.placeholder(tf.int32, [None, 1])
arg_control_group_id_output = tf.placeholder(tf.float32, [None, 1])
arg_select_point_act_output = tf.placeholder(tf.int32, [None, 1])
arg_select_add_output = tf.placeholder(tf.int32, [None, 1])
arg_select_unit_act_output = tf.placeholder(tf.int32, [None, 1])
arg_select_unit_id_output = tf.placeholder(tf.float32, [None, 1])
arg_select_worker_output = tf.placeholder(tf.int32, [None, 1])
arg_build_queue_id_output = tf.placeholder(tf.float32, [None, 1])
arg_unload_id_output = tf.placeholder(tf.float32, [None, 1])

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
minimap_output = tf.layers.dense(dense_minimap, 64)

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

##### arg_type output loss
# screen
screen_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
screen_output_pred = tf.layers.dense(screen_output_dense, 2)
screen_output_loss = tf.reduce_mean(tf.square(screen_output_pred - arg_screen_replay_ouput))
# minimap
minimap_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
minimap_output_pred = tf.layers.dense(minimap_output_dense, 2)
minimap_output_loss = tf.reduce_mean(tf.square(minimap_output_pred - arg_minimap_replay_ouput))
# screen2
screen2_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
screen2_output_pred = tf.layers.dense(screen2_output_dense, 2)
screen2_output_loss = tf.reduce_mean(tf.square(screen2_output_pred - arg_screen2_replay_ouput))
# queued
queued_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
queued_output_logits = tf.layers.dense(queued_output_dense, 2) # enum, [False, True]
queued_pred = tf.nn.softmax(queued_output_logits, name="queued_pred")
queued_pred_cls = tf.argmax(queued_pred, dimension=1)
queued_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=arg_queued_replay_output, 
    logits=queued_output_logits)
queued_loss = tf.reduce_mean(queued_cross_entropy)
# control_group_act
control_group_act_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
control_group_act_logits = tf.layers.dense(control_group_act_output_dense,5) #enum, 5 output, start with 0 
control_group_act_pred = tf.nn.softmax(control_group_act_logits, name="control_group_act_pred")
control_group_act_cls = tf.argmax(control_group_act_pred, dimension=1)
control_group_act_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=arg_control_group_act_replay_output, 
    logits=control_group_act_logits)
control_group_act_loss = tf.reduce_mean(control_group_act_cross_entropy)
# control_group_id
control_group_id_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
control_group_id_output = tf.layers.dense(control_group_id_output_dense, 1)
control_group_id_loss = tf.square(control_group_id_output - arg_control_group_id_output)
# select_point_act
select_point_act_dense = tf.layers.desne(concat_input, 16, tf.nn.relu)
select_point_act_logits = tf.layers.dense(select_point_act_dense, 4) # enum, 4 output
select_point_act_pred = tf.nn.softmax(select_point_act_logits, name="select_point_act_pred")
select_point_act_cls = tf.argmax(select_point_act_pred, dimension=1)
select_point_act_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=arg_select_point_act_output, 
    logits=select_point_act_logits)
select_point_act_loss = tf.reduce_mean(select_point_act_cross_entropy)
# select_add
select_add_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
select_add_output_logits = tf.layers.dense(select_add_output_dense, 2) # enum, [False, True]
select_add_pred = tf.nn.softmax(select_add_output_logits, name="select_add_pred")
select_add_pred_cls = tf.argmax(select_add_pred, dimension=1)
select_add_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=arg_select_add_output, 
    logits=select_add_output_logits)
select_add_loss = tf.reduce_mean(select_add_cross_entropy)
# select_unit_act
select_unit_act_dense = tf.layers.desne(concat_input, 16, tf.nn.relu)
select_unit_act_logits = tf.layers.dense(select_unit_act_dense, 4) # enum, 4 output
select_unit_act_pred = tf.nn.softmax(select_unit_act_logits, name="select_unit_act_pred")
select_unit_act_cls = tf.argmax(select_unit_act_pred, dimension=1)
select_unit_act_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=arg_select_unit_act_output, 
    logits=select_unit_act_logits)
select_unit_act_loss = tf.reduce_mean(select_unit_act_cross_entropy)
# select_unit_id
select_unit_id_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
select_unit_id_output = tf.layers.dense(select_unit_id_output_dense, 1)
select_unit_id_loss = tf.square(select_unit_id_output - arg_select_unit_id_output)
# select_worker
select_worker_dense = tf.layers.desne(concat_input, 16, tf.nn.relu)
select_worker_logits = tf.layers.dense(select_worker_dense, 4) # enum, 4 output
select_worker_pred = tf.nn.softmax(select_worker_logits, name="select_worker_pred")
select_worker_cls = tf.argmax(select_worker_pred, dimension=1)
select_worker_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=arg_select_worker_output, 
    logits=select_worker_logits)
select_worker_loss = tf.reduce_mean(select_worker_cross_entropy)
# build_queue_id
build_queue_id_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
build_queue_id_output = tf.layers.dense(build_queue_id_output_dense, 1)
build_queue_id_loss = tf.square(build_queue_id_output - arg_build_queue_id_output)
# unload_id
unload_id_output_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
unload_id_output = tf.layers.dense(unload_id_output_dense, 1)
unload_id_loss = tf.square(unload_id_output - arg_unload_id_output)

####### Function types for output
Function_type_losses = {
    'move_camera': minimap_output_loss,
    'select_point': select_point_act_loss+screen_output_loss,
    'select_rect': select_add_loss+screen2_output_loss,
    'select_unit': select_unit_act_loss+select_unit_id_loss,
    'control_group': control_group_act_loss+control_group_id_loss,
    'select_idle_worker': select_worker_loss,
    'select_army': select_add_loss,
    'select_warp_gates': select_add_loss,
    'unload': unload_id_loss,
    'build_queue': build_queue_id_loss,
    'cmd_quick': queued_loss,
    'cmd_screen': queued_loss+screen_output_loss,
    'cmd_minimap': queued_loss+minimap_output_loss,
}


train_ops = {}
for key, loss in Function_type_losses.items():
    train_ops[key] = tf.train.AdamOptimizer(LR).minimize(loss)


# regression_dense = tf.layers.dense(concat_input, 16, tf.nn.relu)
# # dropout_regression = tf.layers.dropout(
# #     inputs=dense_screen, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
# regression_output = tf.layers.dense(regression_dense, 2)

# # loss
# loss = tf.reduce_mean(tf.square(regression_output - X_Y_ouput))
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
tf.summary.scalar('loss', loss) # add loss to scalar summary

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph
saver = tf.train.Saver() # define a saver for saving and restoring
writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
merge_op = tf.summary.merge_all() # operation to merge all summary


bg = batchGenerator()
for step in range(5000):                             # train
    m,s,a,u,y,ft =  bg.next_batch_params()

    for i in range(0, len(ft)):
        if ft[i] == 'move_camera':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_minimap_replay_ouput:y[i][0]})
        elif ft[i] == 'select_point':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_select_point_act_output: y[i][0],
                arg_screen_replay_ouput: y[i][1]})
        elif ft[i] == 'select_rect':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_select_add_output: y[i][0],
                arg_screen_replay_ouput: y[i][1],
                arg_screen2_replay_ouput: y[i][2]})
        elif ft[i] == 'select_unit':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_select_unit_act_output: y[i][0],
                arg_select_unit_id_output: y[i][1]})
        elif ft[i] == 'control_group':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_control_group_act_replay_output: y[i][0],
                arg_control_group_id_output: y[i][1]})
        elif ft[i] == 'select_idle_worker':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_select_worker_output: y[i][0]})
        elif ft[i] == 'select_army':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_select_add_output: y[i][0]})
        elif ft[i] == 'select_warp_gates':
             _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_select_add_output: y[i][0]})
        elif ft[i] == 'unload':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_unload_id_output: y[i][0]})
        elif ft[i] == 'build_queue':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_build_queue_id_output: y[i][0]})
        elif ft[i] == 'cmd_quick':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_queued_replay_output: y[i][0]})
        elif ft[i] == 'cmd_screen':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_queued_replay_output: y[i][0],
                arg_screen_replay_ouput: y[i][1]})
        elif ft[i] == 'cmd_minimap':
            _, loss_, result = sess.run([train_ops[ft[i]], Function_type_losses[ft[i]], merge_op],
                {minimap_placeholder: m[i], 
                screen_placeholder: s[i], 
                action_placeholder: a[i], 
                user_info_placeholder:u[i], 
                arg_queued_replay_output: y[i][0],
                arg_minimap_replay_ouput: y[i][1]})
        else:
            print("unknown FUNCTION types !!")        

    if step % 50 == 0:
        m,s,a,u,y,ft =  bg.next_batch_params(get_validation_data=True)
        total_loss = 0
        for i in range(0, len(ft)):
            if ft[i] == 'move_camera':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_minimap_replay_ouput:y[i][0]})
            elif ft[i] == 'select_point':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_select_point_act_output: y[i][0],
                    arg_screen_replay_ouput: y[i][1]})
            elif ft[i] == 'select_rect':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_select_add_output: y[i][0],
                    arg_screen_replay_ouput: y[i][1],
                    arg_screen2_replay_ouput: y[i][2]})
            elif ft[i] == 'select_unit':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_select_unit_act_output: y[i][0],
                    arg_select_unit_id_output: y[i][1]})
            elif ft[i] == 'control_group':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_control_group_act_replay_output: y[i][0],
                    arg_control_group_id_output: y[i][1]})
            elif ft[i] == 'select_idle_worker':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_select_worker_output: y[i][0]})
            elif ft[i] == 'select_army':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_select_add_output: y[i][0]})
            elif ft[i] == 'select_warp_gates':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_select_add_output: y[i][0]})
            elif ft[i] == 'unload':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_unload_id_output: y[i][0]})
            elif ft[i] == 'build_queue':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_build_queue_id_output: y[i][0]})
            elif ft[i] == 'cmd_quick':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_queued_replay_output: y[i][0]})
            elif ft[i] == 'cmd_screen':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_queued_replay_output: y[i][0],
                    arg_screen_replay_ouput: y[i][1]})
            elif ft[i] == 'cmd_minimap':
                loss_, result = sess.run([Function_type_losses[ft[i]], merge_op],
                    {minimap_placeholder: m[i], 
                    screen_placeholder: s[i], 
                    action_placeholder: a[i], 
                    user_info_placeholder:u[i], 
                    arg_queued_replay_output: y[i][0],
                    arg_minimap_replay_ouput: y[i][1]})
            else:
                print("unknown FUNCTION types !!")

            total_loss += loss_

        writer.add_summary(result, step)
        print('step: ', step, 'loss: ',loss_, 'result: ', result)

saver.save(sess, './params', write_meta_graph=False)  # meta_graph is not recommended





