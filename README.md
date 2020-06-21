# utility
**Complexity Estimation utility of neural network**
This utility can be used to estimate how much complexity the neural network will have interms of Multiply and Accumulate (Macs). This helps to designers/developers to quickly gaze whether any Neural network model is able to get fitted in particular edge device .
Just as an e.g : Resnet-50 takes 3.8Giga MACs, hence device chosen should support this much of complexity. 
   Designers/Developers can fit this many operations by using vectorized engines such as NEON/AVX etc.
This is to just give first glance of neural network complexity.

Note : This utility does not include softmax or any activation in computation, to make it generic.
       This only includes CONV2D,BatchNormlization and Dense layers, which contribute major complexity in any neural network.

Quick Start :
            1) git clone https://github.com/vinaybk/utility.git 
            2) Install Tensorflow 1.14 or 2.x , numpy and sys using either conda or pip installs.
            3) In Model folder there is one Resnet-50 model stored. 
            4) Extract Model and place it in folder.
            5) Run the utility as 
                  "python estimate_complexity.py <Model_path>"
                  python estimate_complexity.py Model\resnet-50_keras.h5
            6) This will give each layer Macs and Additions seperately.
            7) At the end it gives Total Macs and Additions.
            8) Glance of ouput may be like this :

---------------------------------------------------------------------------------------------------------------------------------------
Layer Name               Layer Input Shape       Layer Output shape      Weights shape           No of Macs              No of Adds
---------------------------------------------------------------------------------------------------------------------------------------
conv1                    (230, 230, 3)           (112, 112, 64)          (7, 7, 3, 64)           118013952               64
bn_conv1                 (112, 112, 64)          (112, 112, 64)          (64,)                   1605632                 1605632
:
:
bn5c_branch2b            (7, 7, 512)             (7, 7, 512)             (512,)                  50176                   50176
res5c_branch2c           (7, 7, 512)             (7, 7, 2048)            (1, 1, 512, 2048)       51380224                2048
bn5c_branch2c            (7, 7, 2048)            (7, 7, 2048)            (2048,)                 200704                  200704
fc1000                   (None, 2048)            (None, 1000)            (2048, 1000)            2048000                 1000
---------------------------------------------------------------------------------------------------------------------------------------

Total Macs (GMacs): 3.87914752
Total Adds (Madds):  21.201832








