#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o sub_conv2d.cu.cc.o sub_conv2d.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -D_MWAITXINTRIN_H_INCLUDED
/usr/bin/g++-4.8 -std=c++11 -shared -o sub_conv2d.so sub_conv2d.cc sub_conv2d.cu.cc.o  ${TF_CFLAGS[@]} -fPIC -lcuda ${TF_LFLAGS[@]} -O3 -march=native -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda-10.0/targets/x86_64-linux/lib

cp *.so ..
