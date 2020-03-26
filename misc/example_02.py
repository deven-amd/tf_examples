#!/usr/bin/python3

import tensorflow as tf
import os

def setenv_hip_visible_devices(value):
    os.environ["HIP_VISIBLE_DEVICES"]=value


def profile_me(func, logdir_path):
    # start profiling
    tf.python.eager.profiler.start()
    # call the function that we need to profile
    func()
    # stop profiling
    profiler_result = tf.python.eager.profiler.stop()
    # save the profiling result
    tf.python.eager.profiler.save(logdir_path, profiler_result)


def simple_matmul():
    
    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
    b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])

    # `a` * `b`
    # [[ 58,  64],
    #  [139, 154]]
    c = tf.matmul(a, b)




if __name__ == "__main__":
    print("TensorFlow version: ", tf.__version__)
    # setenv_hip_visible_devices("0")
    # profile_me(simple_matmul, "/common/tf_examples/logdir")
