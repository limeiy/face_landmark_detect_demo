# -*- coding: utf-8 -*-

from process_data import *
import tensorflow as tf
import matplotlib.pyplot as plt  # plt用于显示图片
import json
import traceback
import requests

####################################
# this function used the saved model for detection
#####################################
def predict(x_input):
    x_input = np.array(x_input).reshape(-1,96,96,1)
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('model/model.ckpt-10.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
    y_conv = graph.get_tensor_by_name("fc3/y_conv:0")

    print("================debug line==============")
    W_conv1 = graph.get_tensor_by_name('conv1/W_conv1:0')
    weights_conv1 = sess.run(W_conv1)
    print(weights_conv1)
    print("========================================")

    result = sess.run(y_conv, feed_dict={x: x_input, keep_prob:1.0})
    print(result*96)
    print('predict test image done!')

    return result
    # TEST_SIZE = X.shape[0]
    # y_pred = []
    # BATCH_SIZE = 1
    # for j in range(0, TEST_SIZE, BATCH_SIZE):
    #     # y_batch = y_conv.eval(feed_dict={x: X[j:j + BATCH_SIZE], keep_prob: 1.0})
    #     sess.run(y_conv,feed_dict={x:X[j:j+BATCH_SIZE],
    #                                keep_prob:1.0})
    #
    #     y_pred.extend(y_conv)

def plot_images(image, labels):
    plt.imshow(image.reshape(96, 96))
    plt.title("Face Keypoints Detection")
    pointx = labels[::2] * 48 + 48
    pointy = labels[1::2] * 48 + 48
    print(pointx)
    print(pointy)
    plt.scatter(pointx, pointy, marker='x', c='r')  # 画点；'r'红色
    plt.show()

#############################################
#  以下为采用tensorflow serving的方式
#############################################
import sys
import numpy
import threading
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()
    self._predict_result = {}

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

  def record_predict_result(self, no, prediction):
    self._predict_result[no]=prediction

  def get_predict_results(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._predict_result


def _create_rpc_callback(no, result_counter):
  """Creates RPC callback function.
  Args:
    no: The number for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['points'].float_val)
      #prediction = numpy.argmax(response)
      print("prediction is: ", response)

      result_counter.record_predict_result(no, response)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference_grpc(hostport, work_dir, concurrency, images):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  num_tests = images.shape[0]

  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)
  for i in range(num_tests):
    image = images[i]
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tf_face_landmark_detect' # 必须跟serving的MODEL_NAME保持一致
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, image.size]))
    request.inputs['keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))  # 不能直接设置
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(i, result_counter))

  return result_counter.get_predict_results()


def do_inference_model5(hostport, work_dir, concurrency, images):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  num_tests = images.shape[0]

  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)
  for i in range(num_tests):
    image = images[i]
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tf_face_landmark_detect'  ## 要跟serving的MODEL_NAME保持一致
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, image.size]))
    request.inputs['keep_prob1'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))  # 不能直接设置
    request.inputs['keep_prob2'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))
    request.inputs['keep_prob3'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))
    request.inputs['keep_prob4'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, dtype=tf.float32))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(i, result_counter))

  return result_counter.get_predict_results()


def do_inference(hostport, work_dir, concurrency, images):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  num_tests = images.shape[0]

  result_counter = _ResultCounter(num_tests, concurrency)
  for i in range(num_tests):
    image = images[i]
    input_data = {}
    input_data['images'] = image.tolist()
    input_data['keep_prob'] = 1.0

    result_counter.throttle()

    try:
      json_data = {"signature_name": 'predict_images', "instances": [input_data]}
      print(json_data)
      result = requests.post(hostport, json=json_data)
      print(result)
      print(result.text)
    except:
      traceback.print_exc()
      result_counter.inc_error()
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      predict_result = json.loads(result.text)
      print(predict_result)
      prediction = numpy.array(predict_result['predictions'][0])
      print("prediction is: ", prediction)
      result_counter.record_predict_result(i, prediction)
      result_counter.inc_done()
      result_counter.dec_active()

  return result_counter.get_predict_results()

if __name__ == "__main__":

    X, y = read_data(test=True)
    x_input = [X[8]]
    print(x_input)
    labels = predict(x_input)
    plot_images(x_input[0],labels[0])


    # X, result = read_data()
    # plot_images(X[0],result[0])
