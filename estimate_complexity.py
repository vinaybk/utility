import tensorflow 
import tensorflow.keras as keras
#import keras as keras
from tensorflow.keras.models import load_model
import numpy as np
import sys


if len(sys.argv) < 2:
    print("Model path is missing \n usage: python estimate_complexity.py <Model_path>")
    sys.exit()

# check tensorflow version for 2.0
ver = tensorflow.__version__
if(int(ver[0]) == 2):
    Conv_type = tensorflow.compat.v1.keras.layers.Convolution2D
    BN_type = tensorflow.compat.v1.keras.layers.BatchNormalization
    Dense_type = tensorflow.compat.v1.keras.layers.Dense
else:
    Conv_type = keras.layers.Convolution2D
    BN_type = keras.layers.BatchNormalization
    Dense_type =keras.layers.Dense
    
model = load_model(sys.argv[1]) 
i=0
i_bn=0
Quant_size = 0

layer_names = [l.name for l in model.layers]

total_macs = 0
total_adds = 0
print(str('-----').ljust(135,'-'))
print(str("Layer Name").ljust(20),"\t",str('Layer Input Shape').ljust(20),"\t",str('Layer Output shape').ljust(20),"\t",str('Weights shape').ljust(20),"\t",str('No of Macs').ljust(20),"\t",str('No of Adds').ljust(20))
print(str('-----').ljust(135,'-'))
for layer in model.layers:

    op_adds = 0
    op_macs = 0
    
    if type(layer) == Conv_type:  #keras.layers.Convolution2D:
        i=i+1
        
        # convert convolutions ...
        data = layer.get_weights()
        w, b = data if np.shape(data)[0] > 1 else [data[0], np.zeros((1, np.shape(data)[-1]))]  # the bias term might not be present
              
        op_shape = layer.output_shape[1]*layer.output_shape[2] # no_filters are covered by conv2D
        one_ele = w.size
        op_macs = one_ele * op_shape
        op_adds = b.size
        print(str(layer.name).ljust(20),"\t",str(layer.input_shape[1:]).ljust(20),"\t",str(layer.output_shape[1:]).ljust(20),"\t",str(w.shape).ljust(20),"\t",str(op_macs).ljust(20),"\t",str(op_adds).ljust(20))

     
    if type(layer) == BN_type: #keras.layers.BatchNormalization:
        i_bn=i_bn+1
        gamma, beta, mean, variance = layer.get_weights()
   
        op_adds = mean.size*layer.output_shape[1]*layer.output_shape[2] * 2 # 2 additions for mean and bias
        op_macs = mean.size*layer.output_shape[1]*layer.output_shape[2] * 2 # 2 muls for gamma and beta
        print(str(layer.name).ljust(20),"\t",str(layer.input_shape[1:]).ljust(20),"\t",str(layer.output_shape[1:]).ljust(20),"\t",str(mean.shape).ljust(20),"\t",str(op_macs).ljust(20),"\t",str(op_adds).ljust(20))


    if type(layer) ==Dense_type: # keras.layers.Dense:
        i=i+1
        
        # convert convolutions ...
        data = layer.get_weights()
        w, b = data 
        
        bias = b
        op_macs = w.size
        op_adds = b.size
        print(str(layer.name).ljust(20),"\t",str(layer.input_shape).ljust(20),"\t",str(layer.output_shape).ljust(20),"\t",str(w.shape).ljust(20),"\t",str(op_macs).ljust(20),"\t",str(op_adds).ljust(20))
    total_adds = total_adds + op_adds
    total_macs = total_macs + op_macs
print(str('-----').ljust(135,'-'))
print('\nTotal Macs (GMacs):'.ljust(20),str(total_macs/1e09).ljust(5))
print("Total Adds (Madds):".ljust(20),str(total_adds/1e06).ljust(5))
        
        
