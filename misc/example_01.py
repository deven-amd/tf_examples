#!/usr/bin/python3

import tensorflow as tf
import os

##
##
def setenv_hip_visible_devices(value):
    os.environ["HIP_VISIBLE_DEVICES"]=value

##
##
def get_all_physical_devices():
    return tf.config.experimental.list_physical_devices()

def get_all_logical_devices():
    return tf.config.experimental.list_logical_devices()

def get_cpu_physical_devices():        
    return tf.config.experimental.list_physical_devices("CPU")

def get_cpu_logical_devices():        
    return tf.config.experimental.list_logical_devices("CPU")

def get_gpu_physical_devices():        
    return tf.config.experimental.list_physical_devices("GPU")

def get_gpu_logical_devices():        
    return tf.config.experimental.list_logical_devices("GPU")

##
##
def set_visible_gpu_devices():
    gpus = get_gpu_physical_devices()
    tf.config.experimental.set_visible_devices(gpus[0:1], 'GPU')

##
##
def list_all_physical_devices():
    print ("-"*50, "ALL Physical Devices")
    for device in get_all_physical_devices():
        print (device)
    print ("-"*50)

def list_all_logical_devices():
    print ("-"*50, "ALL Logical Devices")
    for device in get_all_logical_devices():
        print (device)
    print ("-"*50)

def list_cpu_physical_devices():        
    print ("-"*50, "CPU Physical Devices")
    for device in get_cpu_physical_devices():
        print (device)
    print ("-"*50)

def list_cpu_logical_devices():        
    print ("-"*50, "CPU Logical Devices")
    for device in get_cpu_logical_devices():
        print (device)
    print ("-"*50)

def list_gpu_physical_devices():        
    print ("-"*50, "GPU Physical Devices")
    for device in get_gpu_physical_devices():
        print (device)
    print ("-"*50)

def list_gpu_logical_devices():        
    print ("-"*50, "GPU Logical Devices")
    for device in get_gpu_logical_devices():
        print (device)
    print ("-"*50)

##
##
def create_two_logical_gpus_on_one_physical_gpu():
    setenv_hip_visible_devices("0")
    gpus = get_gpu_physical_devices()
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    list_gpu_logical_devices()

    
##
##
def display_device_placement():
    tf.debugging.set_log_device_placement(True)

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    
    print(c)

##
##
if __name__ == "__main__":
    create_two_logical_gpus_on_one_physical_gpu()
    
    

    


    
    
