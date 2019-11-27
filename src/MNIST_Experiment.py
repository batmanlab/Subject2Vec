"""Final version of DeepSet with encoder+ decoder + attention.

We have cross entropy over the digits. It required true lables i.e prime or not
But we are not using it in loss function hence not optimizing over it.

The cross entropy is just for exploratory analysis.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pdb
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime
import os
import h5py
from sklearn.preprocessing import StandardScaler
import math 
from COPD_Sub2Vec_Helper import encoder, decoder, discriminativeNetwork


#Parse the arguments
parser = argparse.ArgumentParser(description='Subject2Vector on MNIST')
parser.add_argument('--data_dir', type=str, default='MNIST_Data/', help='Directory where MNIST data is stored')
parser.add_argument('--experiment_name', type=str, default='new_MNIST_Exp', help='Name of the experiment')
parser.add_argument('--no_of_epoch', type=int, default=40,
                  help='Number of epochs for training.')
parser.add_argument('--batch_size', type=int, default=1,
                  help='Number of sets to be processed in a batch.')
parser.add_argument('--max_set_size', type=int, default=100,
                  help='Maximum size of any set.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
parser.add_argument( '--log_dir', type=str, default='../Output/Log', help='Summaries log directory')
parser.add_argument( '--checkpoint_dir', type=str, default='../Output/Checkpoint', help='Checkpoint Saved')
parser.add_argument("--seed", type=int, default=547, help='Seed for tensorflow and numpy.')
parser.add_argument('--lambda1', type=float, default=0.001, help='Paramter to balance generative loss.')
parser.add_argument('--lambda2', type=float, default=0.001, help='Paramter to balance regularization loss.')
parser.add_argument( '--train_batch', type=int, default=1000, help='Number of batches in training.')
parser.add_argument( '--test_batch', type=int, default=300, help='Number of batches in testing.')
parser.add_argument( '--val_batch', type=int, default=500, help='Number of batches in validation.')
parser.add_argument('--past_checkpoint', type=str, default=None, help='Name of the past checkpoint expeiment from where to continue training')

a = parser.parse_args()  
log_dir = os.path.join(a.log_dir, a.experiment_name)
checkpoint_dir = os.path.join(a.checkpoint_dir, a.experiment_name)

#Golbal Variables
_index_in_epoch = 0

mnist = None
num_classes = 2 #Prime or not


def R2(tv,pv):
        res = np.sum(np.square(tv-pv))
        var = np.sum(np.square(tv-np.mean(tv)))
        return 1 - (res / var)

def train():
    sess = tf.InteractiveSession()
    # Import data
    
    acc = np.zeros([3, a.no_of_epoch])
    lossT = np.zeros([3, a.no_of_epoch])
    lossG = np.zeros([3, a.no_of_epoch]) #Generative
    lossD = np.zeros([3, a.no_of_epoch]) #Discriminative
    lossR = np.zeros([3, a.no_of_epoch]) #Regularization
    ce1_result = np.zeros([3, a.no_of_epoch])
    ce2_result = np.zeros([3, a.no_of_epoch])
    RSquare = np.zeros([3, a.no_of_epoch])

    # Create a multilayer model.

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input') #[b*s, 784]
        labels = tf.placeholder(tf.float32, [None, num_classes], name='labels') #[b*s, 10]
        y_ = tf.placeholder(tf.float32, [a.batch_size,1], name='y-input') #[b, 1]
        isTrain = tf.placeholder(tf.bool) 
        n = tf.placeholder(tf.int32)
    with tf.name_scope('input_x_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1]) #[Nb, 32, 32, 32, 1]
    H = encoder(x_image, MNIST=True)#[b*s, 1, 1, 128]
    X_Generated = decoder(H, MNIST=True) #[b*s, 784]   
    X_Generated = tf.reshape(X_Generated, [-1, 784])
    #y_predicted, alpha, H_logit_opt1, H_logit_opt2, z 
    nFeatures = H.get_shape().as_list()[1:]
    nFeatures = np.prod(np.array(nFeatures)) #128   
    H_logit_opt2 = tf.reshape(H, [-1, nFeatures]) #[Nb, 1*1*1*128]
    
    y_predicted, alpha, H_logit_opt1= discriminativeNetwork(H, isTrain, n, a.dropout, a.seed, outputDim=1)

    #Cross Enthropy    
    logits_1 = tf.layers.dense(inputs=H_logit_opt1, units=num_classes)
    crossEntropy_1 = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_1)
    logits_2 = tf.layers.dense(inputs=H_logit_opt2, units=num_classes)
    crossEntropy_2 = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_2)
    
    with tf.name_scope('Loss'):
        loss_generative = tf.reduce_mean((x - X_Generated)**2)  #L2 loss function       
        loss_discriminative = tf.reduce_sum(tf.reduce_mean((y_ - y_predicted)**2, axis=0))#L2 
        #Regularization
        epsilon = 0.0001
        alpha_epsilon = tf.add(alpha, epsilon)
        log_alpha = tf.log(alpha_epsilon)
        loss_regularization = tf.reduce_sum(log_alpha)  
        print("lambda2: ", a.lambda2)
        print("lambda1: ", a.lambda1)
        loss = loss_generative  + tf.scalar_mul(a.lambda1,loss_discriminative) + tf.scalar_mul(a.lambda2, loss_regularization) 
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(a.learning_rate).minimize(loss, colocate_gradients_with_ops=True)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):  
            correct_prediction = tf.less_equal(tf.abs(y_ - y_predicted),0.5)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    if a.past_checkpoint is not None:
        past_checkpoint_dir = os.path.join(a.checkpoint_dir, a.past_checkpoint)
        ckpt = tf.train.get_checkpoint_state(past_checkpoint_dir+'/')
        if ckpt and ckpt.model_checkpoint_path: 
            print("HERE............................")
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, tf.train.latest_checkpoint(past_checkpoint_dir+'/'))
            f1 = open(ckpt.model_checkpoint_path + '.txt', 'w')
            f1.write(ckpt.model_checkpoint_path)
            f1.close
        else:
            f1 = open(a.past_checkpoint + '.txt', 'w')
            f1.write("Nothing")
            f1.close
            pdb.set_trace()
            
    
    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = a.test_batch
    val_batches_per_epoch = a.val_batch
    test_batches_per_epoch = a.test_batch
    
    # Train the model,
    for epoch in range(a.no_of_epoch):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        train_accuracy = 0.0
        total_loss = 0.0
        loss_g = 0.0
        loss_d = 0.0
        loss_r = 0.0
        train_ce_1 = 0.0
        train_ce_2 = 0.0
        TrueValue = []
        PredictedValue = []
        for i in range(train_batches_per_epoch):
            #xs: images
            #ys: Sum pf prime numbers in the set
            #yys: binary labels for elements of set (prime or not)
            #yyys: One hot representation of actual number associated with the image in set
            xs, ys, yys, yyys= get_batch(train=1)  
            [_,_accuracy, _loss_generative, _loss_discriminative, _loss_regularization, _loss, ce1, ce2, _y_predicted] = sess.run([train_step,accuracy, loss_generative, loss_discriminative, loss_regularization, loss, crossEntropy_1, crossEntropy_2, y_predicted], feed_dict={x:xs, isTrain:True, y_: ys, n:xs.shape[0],labels:yys}) 
            train_accuracy += _accuracy  
            total_loss += _loss
            train_ce_1 += ce1
            train_ce_2 += ce2
            loss_g += _loss_generative
            loss_d += _loss_discriminative
            loss_r += _loss_regularization
        
            for v in ys:
                TrueValue.append(v)
            for v in _y_predicted:
                PredictedValue.append(v)
        train_accuracy /= train_batches_per_epoch
        total_loss /= train_batches_per_epoch
        train_ce_1 /= train_batches_per_epoch
        train_ce_2 /= train_batches_per_epoch
        loss_g /= train_batches_per_epoch
        loss_d /= train_batches_per_epoch
        loss_r /= train_batches_per_epoch
        TrueValue = np.asarray(TrueValue)
        PredictedValue = np.asarray(PredictedValue)
        acc[0][epoch] = train_accuracy
        lossT[0][epoch] = total_loss
        lossG[0][epoch] =  loss_g#Generative
        lossD[0][epoch] =  loss_d#Discriminative
        lossR[0][epoch] =  loss_r#Regularization
        ce1_result[0][epoch] = train_ce_1
        ce2_result[0][epoch] = train_ce_2
        RSquare[0][epoch] = R2(TrueValue,PredictedValue)
        print("{} Training loss = {:.4f}".format(datetime.now(), total_loss)) 

    # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now())) 
        train_accuracy = 0.0
        total_loss = 0.0
        loss_g = 0.0
        loss_d = 0.0
        loss_r = 0.0
        train_ce_1 = 0.0
        train_ce_2 = 0.0
        TrueValue = []
        PredictedValue = []
        for i in range(val_batches_per_epoch):
            #xs: images
            #ys: Sum pf prime numbers in the set
            #yys: binary labels for elements of set (prime or not)
            #yyys: One hot representation of actual number associated with the image in set
            xs, ys, yys, yyys= get_batch(train=2)                 
            [_accuracy, _loss_generative, _loss_discriminative, _loss_regularization, _loss, ce1, ce2, _y_predicted] = sess.run([accuracy, loss_generative, loss_discriminative, loss_regularization, loss, crossEntropy_1, crossEntropy_2, y_predicted], feed_dict={x:xs, isTrain:False, y_: ys, n:xs.shape[0],labels:yys}) 

            
            train_accuracy += _accuracy  
            total_loss += _loss
            train_ce_1 += ce1
            train_ce_2 += ce2
            loss_g += _loss_generative
            loss_d += _loss_discriminative
            loss_r += _loss_regularization
        
            for v in ys:
                TrueValue.append(v)
            for v in _y_predicted:
                PredictedValue.append(v)
        train_accuracy /= train_batches_per_epoch
        total_loss /= train_batches_per_epoch
        train_ce_1 /= train_batches_per_epoch
        train_ce_2 /= train_batches_per_epoch
        loss_g /= train_batches_per_epoch
        loss_d /= train_batches_per_epoch
        loss_r /= train_batches_per_epoch
        TrueValue = np.asarray(TrueValue)
        PredictedValue = np.asarray(PredictedValue)
        acc[1][epoch] = train_accuracy
        lossT[1][epoch] = total_loss
        lossG[1][epoch] =  loss_g#Generative
        lossD[1][epoch] =  loss_d#Discriminative
        lossR[1][epoch] =  loss_r#Regularization
        ce1_result[1][epoch] = train_ce_1
        ce2_result[1][epoch] = train_ce_2
        RSquare[1][epoch] = R2(TrueValue,PredictedValue)
        print("{} Validation Loss = {:.4f}".format(datetime.now(), total_loss)) 
        
        #print("{} Saving checkpoint of model...".format(datetime.now()))  
    
        #Run testing if valiation accuracy is very high
        if epoch%2 == 0 or epoch == a.no_of_epoch-1:
            train_accuracy = 0.0
            total_loss = 0.0
            loss_g = 0.0
            loss_d = 0.0
            loss_r = 0.0
            train_ce_1 = 0.0
            train_ce_2 = 0.0
            TrueValue = []
            PredictedValue = []
            for i in range(test_batches_per_epoch):
                #xs: images
                #ys: Sum pf prime numbers in the set
                #yys: binary labels for elements of set (prime or not)
                #yyys: One hot representation of actual number associated with the image in set
                xs, ys, yys, yyys= get_batch(train=3)                 
                [_accuracy, _loss_generative, _loss_discriminative, _loss_regularization, _loss, ce1, ce2, _y_predicted] = sess.run([accuracy, loss_generative, loss_discriminative, loss_regularization, loss, crossEntropy_1, crossEntropy_2, y_predicted], feed_dict={x:xs, isTrain:False, y_: ys, n:xs.shape[0],labels:yys}) 


                train_accuracy += _accuracy  
                total_loss += _loss
                train_ce_1 += ce1
                train_ce_2 += ce2
                loss_g += _loss_generative
                loss_d += _loss_discriminative
                loss_r += _loss_regularization

                for v in ys:
                    TrueValue.append(v)
                for v in _y_predicted:
                    PredictedValue.append(v)
            train_accuracy /= train_batches_per_epoch
            total_loss /= train_batches_per_epoch
            train_ce_1 /= train_batches_per_epoch
            train_ce_2 /= train_batches_per_epoch
            loss_g /= train_batches_per_epoch
            loss_d /= train_batches_per_epoch
            loss_r /= train_batches_per_epoch
            TrueValue = np.asarray(TrueValue)
            PredictedValue = np.asarray(PredictedValue)
            acc[2][epoch] = train_accuracy
            lossT[2][epoch] = total_loss
            lossG[2][epoch] =  loss_g#Generative
            lossD[2][epoch] =  loss_d#Discriminative
            lossR[2][epoch] =  loss_r#Regularization
            ce1_result[2][epoch] = train_ce_1
            ce2_result[2][epoch] = train_ce_2
            RSquare[2][epoch] = R2(TrueValue,PredictedValue)
            print("{} Test Loss = {:.4f}".format(datetime.now(), total_loss)) 
            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name) 
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
            np.save(os.path.join(log_dir, 'loss.npy'), lossT)
            np.save(os.path.join(log_dir, 'lossR.npy'), lossR)
            np.save(os.path.join(log_dir, 'lossD.npy'), lossD)
            np.save(os.path.join(log_dir, 'lossG.npy'), lossG)  
            np.save(os.path.join(log_dir, 'accuracy.npy'), acc)
            np.save(os.path.join(log_dir, 'ce1.npy'), ce1_result)
            np.save(os.path.join(log_dir, 'ce2.npy'), ce2_result)
            np.save(os.path.join(log_dir, 'RSquare.npy'), RSquare)
            #Save the graph
            graph_def = tf.get_default_graph().as_graph_def()
            graph_txt = str(graph_def)
            with open(os.path.join(log_dir, 'graph.txt'), 'wt') as f: f.write(graph_txt)

def get_batch(train=1):
    global a
    global mnist
    #Sample the number of elements in a set
    set_size = np.random.randint(20, a.max_set_size, a.batch_size)
    num_elements = np.sum(set_size)
    
    if train == 1:  
        scaler = StandardScaler().fit(mnist.train.images)
        xs, ys = mnist.train.next_batch(num_elements)        
    elif train == 2: #Validation
        scaler = StandardScaler().fit(mnist.validation.images)
        xs, ys = mnist.validation.next_batch(num_elements)
    elif train == 3: #Test
        scaler = StandardScaler().fit(mnist.test.images)
        xs, ys = mnist.test.next_batch(num_elements)
    xs = scaler.transform(xs)

    for i in range(a.batch_size):
        current_Labels =  np.argmax(ys,axis=1)
        s_label = 0
        prime_label = []
        for label in current_Labels:
            if label in [2,3,5,7]:
                s_label += label   #Sum of prime numbers 
                prime_label.append([0,1])
            else:
                prime_label.append([1,0])
    setLabel = np.asarray([s_label])
    setLabel = np.reshape(setLabel, [1,-1])
    yys = np.asarray(prime_label)
    #xs: images
    #setLabel: Sum pf prime numbers in the set
    #yys: binary labels for elements of set (prime or not)
    #ys: One hot representation of actual number associated with the image in set
    return xs, setLabel, yys, ys

def main():
    global log_dir 
    global checkpoint_dir 
    global a
    global mnist
    
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)

    while tf.gfile.Exists(log_dir):
        log_dir = log_dir + "_1"
    tf.gfile.MakeDirs(log_dir)
    while tf.gfile.Exists(checkpoint_dir):
        checkpoint_dir = checkpoint_dir + '_1'
    tf.gfile.MakeDirs(checkpoint_dir)
    
    mnist = input_data.read_data_sets(a.data_dir, one_hot=True)
    
    
    train()


if __name__ == '__main__':
    main()
    
    
