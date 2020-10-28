#!/usr/bin/env python3

import tensorflow as tf
print (tf.__version__)


@tf.function
def legacy_func(x1, x2, x3):
  t1 = tf.math.add(x1, x2)
  t2 = tf.nn.relu(t1)
  return t2
  # t3 = tf.linalg.matmul(t2, x3)
  # t4 = tf.math.reduce_max(t3)
  # return t4

@tf.function(experimental_compile=True)
def xla_func(x1, x2, x3):
  t1 = tf.math.add(x1, x2)
  t2 = tf.nn.relu(t1)
  return t2
  # t3 = tf.linalg.matmul(t2, x3)
  # t4 = tf.math.reduce_max(t3)
  # return t4


def simple_run(x1, x2, x3):
  y1 = legacy_func(x1, x2, x3)
  y2 = xla_func(x1, x2, x3)


def profiling_run(x1, x2):
  tf.profiler.experimental.start('logdir')
  for step in range(100):
    with tf.profiler.experimental.Trace("Train", step_num=step):
      y1 = legacy_func(x1, x2)
      y2 = xla_func(x1, x2)
  tf.profiler.experimental.stop()


def main():

  dtype = tf.dtypes.float16
  input_shape_1 = (1023, 15)
  input_shape_2 = (15, 17)
  
  x1 = tf.random.uniform(input_shape_1, dtype=dtype)
  x2 = tf.random.uniform(input_shape_1, dtype=dtype)
  x3 = tf.random.uniform(input_shape_2, dtype=dtype)

  simple_run(x1, x2, x3)

  # profiling_run(x1, x2)
    
  tf.print("All Done!")

  
if __name__ == '__main__':
  main()
