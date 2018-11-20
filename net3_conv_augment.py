# -*- coding: utf-8 -*-
##################################
# 此模型采用传统神经网络，且只有一个隐层
##################################
import tensorflow as tf
from process_data import *
import os

from tensorflow.python import debug as tf_debug

VALIDATION_SIZE = 100
BATCH_SIZE=1024

tf.app.flags.DEFINE_integer('epoch',3000,'training iteration')
tf.app.flags.DEFINE_integer('validation_size',100, 'validation set size')
tf.app.flags.DEFINE_string('model_export_dir', './net3/exported_model', 'Direcotry to save exported model')
tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')
tf.app.flags.DEFINE_string('model_save_dir', './net3/saved_model', 'Direcotry to save model')
tf.app.flags.DEFINE_string('model_save_name', 'model.ckpt',
                            'Name for the saved model')
tf.app.flags.DEFINE_integer('patience', 100, 'Most wait times to stop training if valid loss is not improved')

FLAGS=tf.app.flags.FLAGS

def weight_variable(shape,name=None):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name == None:
    return tf.Variable(initial)
  return tf.Variable(initial,name=name)


def bias_variable(shape,name=None):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  if name == None:
    return tf.Variable(initial)
  return tf.Variable(initial,name=name)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def deepnn(x):
    # 定义网络
    # kaggle提供的训练图片是96×96的灰度图片
    with tf.name_scope('reshape'):
      x_image = tf.reshape(x, [-1, 96, 96, 1])
      tf.summary.image('image', x_image, max_outputs=10)

    with tf.name_scope('conv1'):
      W_conv1 = weight_variable([3, 3, 1, 32], name='W_conv1')
      b_conv1 = bias_variable([32], name='b_conv1')
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      tf.summary.histogram('W_conv1', W_conv1)
      tf.summary.histogram('b_conv1', b_conv1)

    with tf.name_scope('pool1'):
      h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
      W_conv2 = weight_variable([2, 2, 32, 64], name='W_conv2')
      b_conv2 = bias_variable([64], name='b_conv2')
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      tf.summary.histogram('W_conv2', W_conv2)
      tf.summary.histogram('b_conv2', b_conv2)

    with tf.name_scope('pool2'):
      h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
      W_conv3 = weight_variable([2, 2, 64, 128], name='W_conv3')
      b_conv3 = bias_variable([128], name='b_conv3')
      h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
      tf.summary.histogram('W_conv3', W_conv3)
      tf.summary.histogram('b_conv3', b_conv3)

    with tf.name_scope('pool3'):
      h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('fc1'):
      W_fc1 = weight_variable([11 * 11 * 128, 500], name='W_fc1')
      b_fc1 = bias_variable([500], name='b_fc1')

      h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
      tf.summary.histogram('W_fc1', W_fc1)
      tf.summary.histogram('b_fc1', b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([500, 500], name='W_fc2')
        b_fc2 = bias_variable([500], name='b_fc2')
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        tf.summary.histogram('W_fc2', W_fc2)
        tf.summary.histogram('b_fc2', b_fc2)


    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([500, 30], name='W_fc3')
        b_fc3 = bias_variable([30], name='b_fc3')

        y_conv = tf.add(tf.matmul(h_fc2, W_fc3),b_fc3, name='y_conv')
        tf.summary.histogram('W_fc3', W_fc3)
        tf.summary.histogram('b_fc3', b_fc3)


    return y_conv


def main(_):
    best_valid_loss = 1000
    best_valid_epoch = 0

    x = tf.placeholder(tf.float32, [None,96*96], name='x')
    y_ = tf.placeholder(tf.float32, [None, 30])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')        # 实际在这里并没有用，只是为了跟其他模型调用统一参数

    # Build the graph for the net
    y_conv = deepnn(x)

    with tf.name_scope('loss'):
        #loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
        loss = tf.reduce_mean(tf.square(y_ - y_conv))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.InteractiveSession()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # 变量都要初始化
    sess.run(tf.global_variables_initializer())

    #X, y = read_data()
    X,y = data_augment()
    X = X.reshape(-1, 96*96)
    X_valid, y_valid = X[:VALIDATION_SIZE],y[:VALIDATION_SIZE]
    X_train, y_train = X[VALIDATION_SIZE:],y[VALIDATION_SIZE:]

    TRAIN_SIZE = X_train.shape[0]

    train_index = np.arange(TRAIN_SIZE)
    np.random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]

    merged =  tf.summary.merge_all()

    saver = tf.train.Saver()
    print("begin training...Train data size: {0}".format(TRAIN_SIZE))
    fp = open('./net3/net3_loss.txt','w')
    fp.write('train_loss valid_loss\n')
    for i in range(FLAGS.epoch):
        np.random.shuffle(train_index)
        X_train, y_train = X_train[train_index], y_train[train_index]
        for j in range(0, TRAIN_SIZE, BATCH_SIZE):
            #print("Epoch {0}, train {1} samples start...".format(i,j))
            if j+BATCH_SIZE > TRAIN_SIZE:
                end = TRAIN_SIZE
            else:
                end = j+BATCH_SIZE
            train_step.run(feed_dict={x:X_train[j:end],
                                      y_:y_train[j:end],
                                      keep_prob:1.0})


        train_loss = loss.eval(feed_dict={x: X_train, y_: y_train, keep_prob:1.0})
        validation_loss = loss.eval(feed_dict={x: X_valid, y_: y_valid, keep_prob:1.0})
        fp.write(str(train_loss)+" "+str(validation_loss)+"\n")

        print('epoch {0} done! train loss: {1}   validation loss:{2}'.format(i, train_loss, validation_loss ))


        # ====early stopping====
        # i is current epoch
        if validation_loss < best_valid_loss:
            best_valid_epoch = i
            best_valid_loss = validation_loss
        if best_valid_epoch + FLAGS.patience < i:
            print("Early stopping")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                best_valid_loss, best_valid_epoch))
            break


    print("Training Done!")
    fp.close()
    # model_saved_path = os.path.join(FLAGS.model_save_dir, FLAGS.model_save_name)
    # print('Saving trained model to: ', model_saved_path)
    # saver.save(sess, model_saved_path, global_step=FLAGS.epoch)

    # Export model
    export_path_base = FLAGS.model_export_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to: ', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y_conv)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x, 'keep_prob': tensor_info_keep_prob},
            outputs={'points': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature
        }
    )
    builder.save()  # !!! MUST HAVE

    print('Done exporting!')

if __name__=='__main__':
    tf.app.run()