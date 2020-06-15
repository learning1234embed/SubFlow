# [RTAS 2020] SubFlow: A Dynamic Induced-Subgraph Strategy Toward Real-Time DNN Inference and Training
## Introduction

This Git repository provides the source code of ***SubFlow***, an [RTAS 2020](http://2020.rtas.org/) paper titled "***SubFlow: A Dynamic Induced-Subgraph Strategy Toward Real-Time DNN Inference and Training***". SubFlow enables real-time inference and training of a deep neural network (DNN) by dynamically executing a sub-graph of the DNN according to the timing constraint changing at run-time.

This repository generates and executes an example deep neural network (DNN) that is trained and inferred with SubFlow, i.e., [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The following steps show the procedure for the 'SubFlow LeNet-5 DNN', which consists of 1) computation of neuron importance, 2) generation of SubFlow DNN, 3) dynamic real-time training, and 4) dynamic real-time inference. For the reviewers' convenience, we provide a Python script (**subflow.py**) that automatically performs each step with simple command-line options.

## Software Install and Code Cloning
SubFlow is implemented based on Python and TensorFlow. The TensorFlow version should be lower than or equal to 1.13.2; the latest version (1.14) seems to have a problem of executing custom operations. We used Tensorflow 1.13.1 and Python 2.7.

**Step 1.** Install [Python (>= 2.7)](https://www.python.org/downloads/).

**Step 2.** Install [Tensorflow (<= 1.13.2)](https://www.tensorflow.org/).

**Step 3.** Install [NVIDIA CUDA (>= 10.0)](https://developer.nvidia.com/cuda-downloads).

**Step 4.** Clone this SubFlow repository.
```sh
$ git clone https://github.com/realtimednn-rtas2020/SubFlow.git
Cloning into 'SubFlow'...
remote: Enumerating objects: 47, done.
remote: Counting objects: 100% (47/47), done.
remote: Compressing objects: 100% (38/38), done.
remote: Total 47 (delta 21), reused 23 (delta 8), pack-reused 0
Unpacking objects: 100% (47/47), done.
```

## 0) Build and Train a Target DNN (Preliminary)
Before getting into the actual SubFlow, a target DNN (i.e., LeNet-5) that we want to apply SubFlow should be first created and trained. Although this preliminary step does not belong to the procedure of SubFlow, we include it here for a demonstration purpose.  

**Step 0-1.** Create a target DNN with TensorFlow by using the script (**subflow.py**). The '-layers' option specifies the architecture of the DNN.
```sh
$ python subflow.py -mode=c -layers='28*28*1,5*5*1*6,5*5*6*16,400,84,10'
[c] constructing a network
[c] layers: 28*28*1,5*5*1*6,5*5*6*16,400,84,10
layer_type: ['input', 'conv', 'max_pool', 'conv', 'max_pool', 'hidden', 'hidden', 'output']
num_of_neuron_per_layer: [[28, 28, 1], [12, 12, 6], [4, 4, 16], [400], [84], [10]]
num_of_neuron_per_layer_without_pool: [[28, 28, 1], [24, 24, 6], [8, 8, 16], [400], [84], [10]]
num_of_weight_per_layer: [150, 2400, 102400, 33600, 840]
num_of_bias_per_layer: [6, 16, 400, 84, 10]
Tensor("neuron_0:0", shape=(?, 28, 28, 1), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_0:0' shape=(5, 5, 1, 6) dtype=float32_ref>, 'biases': <tf.Variable 'bias_0:0' shape=(6,) dtype=float32_ref>}
Tensor("neuron_1:0", shape=(?, 12, 12, 6), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_1:0' shape=(5, 5, 6, 16) dtype=float32_ref>, 'biases': <tf.Variable 'bias_1:0' shape=(16,) dtype=float32_ref>}
LAST CNN BEFORE FC
Tensor("neuron_2:0", shape=(?, 16, 4, 4), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_2:0' shape=(256, 400) dtype=float32_ref>, 'biases': <tf.Variable 'bias_2:0' shape=(400,) dtype=float32_ref>}
Tensor("neuron_3:0", shape=(?, 400), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_3:0' shape=(400, 84) dtype=float32_ref>, 'biases': <tf.Variable 'bias_3:0' shape=(84,) dtype=float32_ref>}
Tensor("neuron_4:0", shape=(?, 84), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_4:0' shape=(84, 10) dtype=float32_ref>, 'biases': <tf.Variable 'bias_4:0' shape=(10,) dtype=float32_ref>}
Tensor("neuron_5:0", shape=(?, 10), dtype=float32)
```

**Step 0-2.** Train the newly-created DNN with TensorFlow.
```sh
$ python subflow.py -mode=t -network_no=1 -data=mnist_data
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
[t] train
[t] network_no: 1
[t] data: mnist_data train/test.size: (55000, 784) (10000, 784)
train
step 0, training accuracy: 0.104167
step 0, Validation accuracy: 0.100500
step 100, training accuracy: 0.937500
step 100, Validation accuracy: 0.924400
saveNetwork for 0.9244
step 200, training accuracy: 0.937500
step 200, Validation accuracy: 0.953800
saveNetwork for 0.9538
...
...
...
step 1800, training accuracy: 0.989583
step 1800, Validation accuracy: 0.987400
saveNetwork for 0.9874
step 1900, training accuracy: 0.989583
step 1900, Validation accuracy: 0.986300
step 1999, training accuracy: 0.989583
step 1999, Validation accuracy: 0.986600
```

## 1) Computation of Neuron Importance (Offline)
After obtaining a target DNN, the importance of the neurons of the target DNN is computed to rank the neurons, i.e., which neuron is more important than others to performance (i.e., inference accuracy) of the DNN. The neuron importance is calculated layer by layer using the training data samples.

**Step 1-1.** Compute and save the neuron importance.
```sh
$ python subflow.py -mode=ci -network_no=1 -data=mnist_data
[ci] update importance
[ci] network_no: 1
[ci] data: mnist_data train/test.size: (55000, 784) (10000, 784)
update_importance
sample 0
Tensor("mul_2:0", shape=(?, 28, 28, 1), dtype=float32)
Tensor("mul_3:0", shape=(?, 24, 24, 6), dtype=float32)
Tensor("mul_4:0", shape=(?, 8, 8, 16), dtype=float32)
Tensor("mul_5:0", shape=(?, 400), dtype=float32)
Tensor("mul_6:0", shape=(?, 84), dtype=float32)
Tensor("mul_7:0", shape=(?, 10), dtype=float32)
sample 1
Tensor("mul_8:0", shape=(?, 28, 28, 1), dtype=float32)
Tensor("mul_9:0", shape=(?, 24, 24, 6), dtype=float32)
Tensor("mul_10:0", shape=(?, 8, 8, 16), dtype=float32)
Tensor("mul_11:0", shape=(?, 400), dtype=float32)
Tensor("mul_12:0", shape=(?, 84), dtype=float32)
Tensor("mul_13:0", shape=(?, 10), dtype=float32)
...
...
...
sample 19
Tensor("mul_116:0", shape=(?, 28, 28, 1), dtype=float32)
Tensor("mul_117:0", shape=(?, 24, 24, 6), dtype=float32)
Tensor("mul_118:0", shape=(?, 8, 8, 16), dtype=float32)
Tensor("mul_119:0", shape=(?, 400), dtype=float32)
Tensor("mul_120:0", shape=(?, 84), dtype=float32)
Tensor("mul_121:0", shape=(?, 10), dtype=float32)
saving new importance
```

## 2) Generation of SubFlow DNN (Offline)
The fully-trained target DNN (i.e., LeNet-5) of which neuron importance has been computed is transformed into a SubFlow DNN. The transformation can be easily performed by simply applying SubFlow operations to the DNN model (i.e., convolution and matrix multiplication). For simplicity of implementation, we create a new SubFlow DNN having the identical architecture of the target DNN. The only difference between them is the operations used for building the two DNN models (TensorFlow operations vs. SubFlow operations).

**Step 2-1.** Construct a SubFlow DNN from the target DNN by applying the SubFlow operations.
```sh
$ python subflow.py -mode=sc -network_no=1 
[sc] constructing a sub_network
[sc] network_no: 1
layer_type: ['input', 'conv', 'max_pool', 'conv', 'max_pool', 'hidden', 'hidden', 'output']
num_of_neuron_per_layer: [[28, 28, 1], [12, 12, 6], [4, 4, 16], [400], [84], [10]]
num_of_neuron_per_layer_without_pool: [[28, 28, 1], [24, 24, 6], [8, 8, 16], [400], [84], [10]]
num_of_weight_per_layer: [150, 2400, 102400, 33600, 840]
num_of_bias_per_layer: [6, 16, 400, 84, 10]
Tensor("neuron_0:0", shape=(?, 1, 28, 28), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_0:0' shape=(6, 1, 5, 5) dtype=float32_ref>, 'biases': <tf.Variable 'bias_0:0' shape=(6,) dtype=float32_ref>}
output_biased Tensor("Mul:0", shape=(?, 6, 24, 24), dtype=float32)
Tensor("neuron_1:0", shape=(?, 6, 12, 12), dtype=float32)
conv_parameter {'weights': <tf.Variable 'weight_1:0' shape=(16, 6, 5, 5) dtype=float32_ref>, 'biases': <tf.Variable 'bias_1:0' shape=(16,) dtype=float32_ref>}
output_biased Tensor("Mul_1:0", shape=(?, 16, 8, 8), dtype=float32)
Tensor("neuron_2:0", shape=(?, 16, 4, 4), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_2:0' shape=(256, 400) dtype=float32_ref>, 'biases': <tf.Variable 'bias_2:0' shape=(400,) dtype=float32_ref>}
Tensor("neuron_3:0", shape=(?, 400), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_3:0' shape=(400, 84) dtype=float32_ref>, 'biases': <tf.Variable 'bias_3:0' shape=(84,) dtype=float32_ref>}
Tensor("neuron_4:0", shape=(?, 84), dtype=float32)
fc_parameter {'weights': <tf.Variable 'weight_4:0' shape=(84, 10) dtype=float32_ref>, 'biases': <tf.Variable 'bias_4:0' shape=(10,) dtype=float32_ref>}
Tensor("neuron_5:0", shape=(?, 10), dtype=float32)
```

## 3) Dynamic Real-Time Training (Online)
With the SubFlow DNN being created, dynamic execution (both inference and training) is ready to go. The SubFlow DNN can be trained with the different network utilization parameters for a training iteration(s), as proposed in the paper. For the demonstration purpose, we randomly select the network utilization between 0.1 and 1.0 and execute the training.

**Step 3-1.** Execute dynamic real-time training with different network utilization. The '-utilization=0' option tells that we use a set of random network utilizations for training.
```sh
$ python subflow.py -mode=st -subflow_network_no=1 -utilization=0 -data=mnist_data
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
[st] sub_train
[st] subflow_network_no: 1
[st] data: mnist_data train/test.size: (55000, 784) (10000, 784)
sub_train
get_activation
utilization 0.800000
do_sub_train
step 0, training accuracy: 1.000000
step 0, Validation accuracy: 0.988600
step 99, training accuracy: 1.000000
step 99, Validation accuracy: 0.990900
saveNetwork for 0.9909
get_activation
utilization 0.600000
do_sub_train
step 0, training accuracy: 0.989583
step 0, Validation accuracy: 0.989500
step 99, training accuracy: 0.979167
step 99, Validation accuracy: 0.989800
saveNetwork for 0.9898
...
...
...
get_activation
utilization 0.200000
do_sub_train
step 0, training accuracy: 0.989583
step 0, Validation accuracy: 0.978400
step 99, training accuracy: 1.000000
step 99, Validation accuracy: 0.980500
saveNetwork for 0.9805
```

## 4) Dynamic Real-Time Inference (Online)
After finishing the training, SubFlow DNN can execute dynamic real-time inference by setting the network utilization parameter. For the demonstration purpose, we execute the inference with the network utilization setting between 0.1 and 1.0.

**Step 4-1.** Execute dynamic real-time inference with different network utilization. The '-utilization=0' option tells that we use the network utilization setting between 0.1 and 1.0 when executing an inference.
```sh
$ python subflow.py -mode=si -subflow_network_no=1 -utilization=0 -data=mnist_data
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
[si] sub_inference
sub_infer
get_activation
utilization 0.100000
Inference accuracy: 0.945600
sub_infer
get_activation
utilization 0.200000
Inference accuracy: 0.978700
sub_infer
get_activation
utilization 0.300000
Inference accuracy: 0.984100
sub_infer
get_activation
utilization 0.400000
Inference accuracy: 0.985800
sub_infer
get_activation
utilization 0.500000
Inference accuracy: 0.987600
sub_infer
get_activation
utilization 0.600000
Inference accuracy: 0.987900
sub_infer
get_activation
utilization 0.700000
Inference accuracy: 0.988700
sub_infer
get_activation
utilization 0.800000
Inference accuracy: 0.989900
sub_infer
get_activation
utilization 0.900000
Inference accuracy: 0.990600
sub_infer
get_activation
utilization 1.000000
Inference accuracy: 0.991800
```

## Execution Time Measurement (Visualization)
By using the Timeline tool of TensorFlow, dynamic execution time of SubFlow DNN with different network utilization settings can be visualized. Our Python script (**subflow.py**) generates a JSON file that traces the execution time of operations for each execution of the SubFlow DNN (inference or training). By parsing the JSON file with Google Chrome web browser, the detailed timeline of SubFlow can be shown.

**Step 1.** Download and launch Google Chrome browser.

**Step 2.** Type 'chrome://tracing/' to the address bar.

**Step 3.** Open a JSON file (ex. infer_1.0.json under 'sub_network1' folder) you want analyze by clicking 'Load' menu.

The following images show the visualized timeline of the SubFlow DNN execution for inference and training (network utilization 0.1 and 1.0), which are displayed on Chrome.

* **Inferecne with network utilization 0.1**
![Inferecne with network utilization 0.1](/images/infer_0.1.png)

* **Inferecne with network utilization 1.0**
![Inferecne with network utilization 1.0](/images/infer_1.0.png)

* **Training with network utilization 0.1**
![Training with network utilization 0.1](/images/train_0.1.png)

* **Training with network utilization 1.0**
![Training with network utilization 1.0](/images/train_1.0.png)


&nbsp;
## Citation (BibTeX)
**SubFlow: A Dynamic Induced-Subgraph Strategy Toward Real-Time DNN Inference and Training**
```
@inproceedings{lee2020subflow,
  title={SubFlow: A Dynamic Induced-Subgraph Strategy Toward Real-Time DNN Inference and Training},
  author={Lee, Seulki and Nirjon, Shahriar},
  booktitle={2020 IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS)},
  pages={15--29},
  year={2020},
  organization={IEEE}
}
```
