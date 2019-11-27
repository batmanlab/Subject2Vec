from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pdb
import tensorflow as tf
import numpy as np
import os 

def conv_layer_with_batch_normalization(_inputs, _filters, _kernel_size, _strides, name='Conv3D', _padding='same', activation_fn=tf.nn.elu, MNIST=False):
    with tf.name_scope(name):
        if not MNIST:
            conv_layer = tf.layers.conv3d(inputs=_inputs, filters=_filters, kernel_size=_kernel_size, strides=_strides, use_bias=False, activation=None, padding=_padding)
        else:
            conv_layer = tf.layers.conv2d(inputs=_inputs, filters=_filters, kernel_size=_kernel_size, strides=_strides, use_bias=False, activation=None, padding=_padding)
        conv_layer = tf.layers.batch_normalization(conv_layer, training=True)
        if activation_fn is not None:
            conv_layer = activation_fn(conv_layer)
        return conv_layer
        
def Deconv_layer_with_batch_normalization(_inputs, _filters, _kernel_size, _strides, activation_fn=tf.nn.elu, name='Deconv3D', _padding='same', stddev=0.02, MNIST=False):
    with tf.variable_scope(name):
        try:
            if not MNIST:
                deconv = tf.layers.conv3d_transpose(inputs=_inputs, filters=_filters, kernel_size=_kernel_size, strides=_strides, padding=_padding, use_bias=False, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=stddev))  
            else:
                deconv = tf.layers.conv2d_transpose(inputs=_inputs, filters=_filters, kernel_size=_kernel_size, strides=_strides, padding=_padding, use_bias=False, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=stddev)) 
            deconv = tf.layers.batch_normalization(deconv, training=True)
            if activation_fn is not None:
                deconv = activation_fn(deconv)
            return deconv
        except AttributeError:
            print(AttributeError)
            pdb.set_trace()
            
def encoder(_inputs, noOfKernels=8, convKernel=3, convStride=1, maxPoolKernel=2, maxPoolStride=2, MNIST=False):    
    #_inputs [Nb, 32, 32, 32, 1]
    print("Encoder Input: ", _inputs.shape)
    
    conv1 = conv_layer_with_batch_normalization(_inputs, noOfKernels, convKernel, convStride, name='conv1',MNIST=MNIST) #[Nb,32,32,32,8] 
    maxPool1 = conv_layer_with_batch_normalization(conv1, noOfKernels, convKernel, maxPoolStride, name='pool1',MNIST=MNIST) #[Nb,16,16,16,8] 
    print("maxPool1:", maxPool1.shape)
    
    conv2 = conv_layer_with_batch_normalization(maxPool1, noOfKernels * 2, convKernel, convStride, name='conv2',MNIST=MNIST) #[Nb,16,16,16,16]
    conv3 = conv_layer_with_batch_normalization(conv2, noOfKernels * 2, convKernel, convStride, name='conv3',MNIST=MNIST) #[Nb,16,16,16,16]
    maxPool2 = conv_layer_with_batch_normalization(conv3, noOfKernels * 2, convKernel, maxPoolStride, name='pool2',MNIST=MNIST) #[Nb,8,8,8,16]
    print("maxPool2:", maxPool2.shape)
    
    conv4 = conv_layer_with_batch_normalization(maxPool2, noOfKernels * 2 * 2, convKernel, convStride, name='conv4',MNIST=MNIST) #[Nb,8,8,8,32]
    conv5 = conv_layer_with_batch_normalization(conv4, noOfKernels * 2 * 2, convKernel, convStride, name='conv5',MNIST=MNIST) #[Nb,8,8,8,32]
    maxPool3 = conv_layer_with_batch_normalization(conv5, noOfKernels * 2 * 2, convKernel, maxPoolStride, name='pool3',MNIST=MNIST) #[Nb,4,4,4,32]
    print("maxPool3: ", maxPool3.shape)
        
    conv6 = conv_layer_with_batch_normalization(maxPool3, noOfKernels * 2 * 2 * 2, convKernel, convStride, name='conv6',MNIST=MNIST)#[Nb,4,4,4,64]
    conv7 = conv_layer_with_batch_normalization(conv6, noOfKernels * 2 * 2 * 2, convKernel, convStride, name='conv7',MNIST=MNIST)#[Nb,4,4,4,64]
    maxPool4 = conv_layer_with_batch_normalization(conv7, noOfKernels * 2 * 2 * 2, convKernel, maxPoolStride, name='pool4',MNIST=MNIST) #[Nb,2,2,2,64]
    print("maxPool4: ", maxPool4.shape)
    
    conv8 = conv_layer_with_batch_normalization(maxPool4, noOfKernels * 2 * 2 * 2 * 2, convKernel, convStride, name='conv8',MNIST=MNIST)#[Nb,2,2,2,512]
    conv9 = conv_layer_with_batch_normalization(conv8, noOfKernels * 2 * 2 * 2 * 2, convKernel, convStride, name='conv9',MNIST=MNIST)#[Nb,2,2,2,128]
    maxPool5 = conv_layer_with_batch_normalization(conv9, noOfKernels * 2 * 2 * 2 * 2, convKernel, maxPoolStride, name='pool5',MNIST=MNIST) #[Nb,1,1,1,128]
    print("maxPool5: ", maxPool5.shape)    
    return maxPool5
    
def decoder(_inputs, noOfKernels=8, convKernel=3, convStride=1, maxPoolKernel=2, maxPoolStride=2, MNIST=False):
    print("Decoder Input: ", _inputs.shape) #x [Nb,1,1,1,128]
    maxPool5_R = Deconv_layer_with_batch_normalization(_inputs, noOfKernels * 16, convKernel, maxPoolStride, name='pool5_R',MNIST=MNIST) #[Nb,[Nb,2,2,2,128]
    conv9_R = Deconv_layer_with_batch_normalization(maxPool5_R, noOfKernels * 2 * 2 * 2 * 2, convKernel, convStride, name='conv9_R',MNIST=MNIST)#[Nb,2,2,2,128]
    conv8_R = Deconv_layer_with_batch_normalization(conv9_R, noOfKernels * 2 * 2 * 2 * 2, convKernel, convStride, name='conv8_R',MNIST=MNIST)#[Nb,2,2,2,512]
    
    maxPool4_R = Deconv_layer_with_batch_normalization(conv8_R, noOfKernels * 2 * 2 * 2, convKernel, maxPoolStride, name='pool4_R',MNIST=MNIST) #[Nb,4,4,4,64]
    print("maxPool4_R", maxPool4_R.shape)
    conv7_R = Deconv_layer_with_batch_normalization(maxPool4_R, noOfKernels * 2 * 2 * 2, convKernel, convStride, name='conv7_R',MNIST=MNIST)#[Nb,4,4,4,64]
    conv6_R = Deconv_layer_with_batch_normalization(conv7_R, noOfKernels * 2 * 2 * 2, convKernel, convStride, name='conv6_R',MNIST=MNIST)#[Nb,4,4,4,64]
    if MNIST:
        maxPool3_R = Deconv_layer_with_batch_normalization(conv6_R, noOfKernels * 2 * 2, convKernel+1, convStride, name='pool3_R' ,_padding ='valid',MNIST=MNIST) #[Nb,7,7,32]
    else:
        maxPool3_R = Deconv_layer_with_batch_normalization(conv6_R, noOfKernels * 2 * 2, convKernel, maxPoolStride, name='pool3_R',MNIST=MNIST) #[Nb,8,8,8,32]
    print("maxPool3_R", maxPool3_R.shape)
    conv5_R = Deconv_layer_with_batch_normalization(maxPool3_R, noOfKernels * 2 * 2, convKernel, convStride, name='conv5_R',MNIST=MNIST) #[Nb,8,8,8,32]
    conv4_R = Deconv_layer_with_batch_normalization(conv5_R, noOfKernels * 2 * 2, convKernel, convStride, name='conv4_R',MNIST=MNIST) #[Nb,8,8,8,32]
    
    maxPool2_R = Deconv_layer_with_batch_normalization(conv4_R, noOfKernels * 2, convKernel, maxPoolStride, name='pool2_R',MNIST=MNIST) #[Nb,16,16,16,16]
    print("maxPool2_R", maxPool2_R.shape)
    conv3_R = Deconv_layer_with_batch_normalization(maxPool2_R, noOfKernels * 2, convKernel, convStride, name='conv3_R',MNIST=MNIST) #[Nb,16,16,16,16]
    conv2_R = Deconv_layer_with_batch_normalization(conv3_R, noOfKernels * 2, convKernel, convStride, name='conv2_R',MNIST=MNIST) #[Nb,16,16,16,16]
    
    maxPool1_R = Deconv_layer_with_batch_normalization(conv2_R, noOfKernels, convKernel, maxPoolStride, name='pool1_r',MNIST=MNIST) #[Nb,32,32,32,8]  
    print("maxPool1_R:", maxPool1_R.shape)
    conv1_R = Deconv_layer_with_batch_normalization(maxPool1_R, 1, convKernel, convStride, name='conv1_R',MNIST=MNIST) #[Nb,32,32,32,1] 
    print("conv1_R:", conv1_R.shape)
    
    return conv1_R

def equivarianceLayer(x, k, n, name='equivarianceLayer'):
    #x: Input [Nb,d]
    #k = output dimension; output [Nb, k]
    #n = size of batch
    #output = sigmoid( bias + (x - (1 (max (x))))W )
    with tf.name_scope(name):
        ones = tf.ones([n,1]) #[Nb, 1]
        x_max = tf.reduce_max(x, axis=0)
        x_max_reshape = tf.reshape(x_max,[1,-1]) #[1,d]
        
        prod = tf.matmul(ones, x_max_reshape) #[Nb, d]        
        x_minus_prod = tf.subtract(x, prod) #[Nb, d]          
        xW = tf.layers.dense(x_minus_prod, k, activation=tf.sigmoid, use_bias=True) #[Nb, k]                      
        return xW
    
def discriminativeNetwork(_inputs, is_training, sizeOfBatch, dropoutRate, seed, outputDim):
    #Here _inputs is the output of the encoder network
    #It is the latent space representation of each patch within a subjectFeature
    
    print("discriminativeNetwork Input: ", _inputs.shape)

    nFeatures = _inputs.get_shape().as_list()[1:]
    nFeatures = np.prod(np.array(nFeatures)) #128   
    print("Dimension of latent space:", nFeatures)
    inputFlat = tf.reshape(_inputs, [-1, nFeatures]) #[Nb, 1*1*1*128]
    
    #Attention Network
    # H [n,d]--> f(H) [n,k]--> z --> softmax --> alpha
    el1 = equivarianceLayer(x=inputFlat, k=64, n=sizeOfBatch, name='EquiVariance1') #[Nb, 64]
    print("el1: ", el1.shape)
    z = equivarianceLayer(x=el1, k=1, n=sizeOfBatch, name='EquiVariance2') #[Nb, 1]
    print("z: ", z.shape)    
    alpha = tf.nn.softmax(z, dim=0, name='softmax') #[Nb, 1]
    print("alpha: ", alpha.shape)
    
    dotProduct = tf.multiply(inputFlat, alpha) #[Nb , 128]
    print("dotProduct: ", dotProduct.shape)
    
    #Rho function
    fc1 = tf.layers.dense(dotProduct, nFeatures, tf.nn.elu, name='fc1') #[Nb, 128] 
    print("fc1_rhoFunction:", fc1.shape)
    fc1_dropout = tf.layers.dropout(fc1, rate=dropoutRate, seed=seed, training=is_training) 
    print("fc1_rhoFunction_dropout", fc1_dropout.shape)
    
    subjectFeature = tf.reduce_sum(fc1_dropout, axis = 0)
    subjectFeatureReshape = tf.reshape(subjectFeature,[1,-1])
    print("subjectFeature: ", subjectFeatureReshape.shape)
    fc2 = tf.layers.dense(subjectFeatureReshape, outputDim, None, name='fc2') #[i, 2]
    return fc2, alpha, fc1_dropout

       
def discriminativeNetwork_confounder(_inputs, is_training, sizeOfBatch, dropoutRate, seed, outputDim, confounder):
    #Here _inputs is the output of the encoder network
    #It is the latent space representation of each patch within a subjectFeature
    
    print("discriminativeNetwork Input: ", _inputs.shape)

    nFeatures = _inputs.get_shape().as_list()[1:]
    nFeatures = np.prod(np.array(nFeatures)) #128   
    print("Dimension of latent space:", nFeatures)
    inputFlat = tf.reshape(_inputs, [-1, nFeatures]) #[Nb, 1*1*1*128]
    
    #Attention Network
    # H [n,d]--> f(H) [n,k]--> z --> softmax --> alpha
    el1 = equivarianceLayer(x=inputFlat, k=64, n=sizeOfBatch, name='EquiVariance1') #[Nb, 64]
    print("el1: ", el1.shape)
    z = equivarianceLayer(x=el1, k=1, n=sizeOfBatch, name='EquiVariance2') #[Nb, 1]
    print("z: ", z.shape)    
    alpha = tf.nn.softmax(z, dim=0, name='softmax') #[Nb, 1]
    print("alpha: ", alpha.shape)
    
    dotProduct = tf.multiply(inputFlat, alpha) #[Nb , 128]
    print("dotProduct: ", dotProduct.shape)
    
    #Rho function
    fc1 = tf.layers.dense(dotProduct, nFeatures, tf.nn.elu, name='fc1') #[Nb, 128] 
    print("fc1_rhoFunction:", fc1.shape)
    fc1_dropout = tf.layers.dropout(fc1, rate=dropoutRate, seed=seed, training=is_training) 
    print("fc1_rhoFunction_dropout", fc1_dropout.shape)
    
    subjectFeature = tf.reduce_sum(fc1_dropout, axis = 0)
    subjectFeatureReshape = tf.reshape(subjectFeature,[1,-1])
    print("subjectFeature: ", subjectFeatureReshape.shape)
    print("confounder: ", confounder.shape)
    final_feature = tf.concat([confounder,subjectFeatureReshape],axis=1)
    print("confounder: ", confounder.shape)
    print("final_feature: ", final_feature.shape)
    fc2 = tf.layers.dense(final_feature, outputDim, None, name='fc2') #[i, 2]
    return fc2, alpha, subjectFeatureReshape            
            
            
            
            
            
            