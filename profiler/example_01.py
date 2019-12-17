#!/usr/bin/python3

import os
os.environ["HIP_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.python.client import timeline

with tf.compat.v1.Session() as sess:
    
    a = tf.random.normal([2000, 5000])
    b = tf.random.normal([5000, 1000])
    res = tf.matmul(a, b)

    # add additional options to trace the session execution
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    sess.run(res, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)
