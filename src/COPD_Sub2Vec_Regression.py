from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse 
import sys
import pdb
import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import os 
import h5py
import pickle
from scipy import interp
from sklearn.tree import DecisionTreeClassifier
from patsy import dmatrices
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy import stats
from COPD_Sub2Vec_Helper import encoder, decoder, discriminativeNetwork_confounder, discriminativeNetwork


parser = argparse.ArgumentParser(description='Subject2Vector')
parser.add_argument('-i', '--input_csv', type=str, default='../Output/Subject2Vec_Input.csv', help='CSV file containing: sid, patch, fold')
parser.add_argument('-ic', '--input_clinicaldata', type=str, default='../Data/Final10000_Phase1_Rev_28oct16.txt', help='CSV file containing the clinical phenotype data from COPDGene dataset')
parser.add_argument('-d', '--data_dir', type=str, default='', help='Directory where input data is stored. Used when "patch" is not an absolute path to the input file.')
parser.add_argument('-cf', '--confounder', type=str, default='',help='A comma separated list of subject-level confounders. e.g. --confounder "Age_Enroll,ATS_PackYears,gender"')
parser.add_argument('-cp', '--clinical_phenotype', type=str, default='FEV1pp_utah,FEV1_FVC_utah',help='A comma separated list of  subject-level labels for the discriminative task. e.g. --clinical_phenotype "FEV1pp_utah,FEV1_FVC_utah"')
parser.add_argument('-t', '--clinical_phenotype_datatype', type=str, default='float-log,float-log',help='A comma separated list of datatype of the clinical phenotypes. e.g. --clinical_phenotype_datatype "float,float". Use "float" for regression, "float-log" for regression on log')
parser.add_argument('-f', '--fold_number', type=int, default=1, help='Number of fold in cross validation.')
parser.add_argument('-m', '--num_gpus', type=int, default=1, help='Number of GPUs.')
parser.add_argument('-n', '--experiment_name', type=str, default='new_Exp', help='Name of the experiment')
parser.add_argument('-c', '--past_checkpoint', type=str, default=None, help='Name of the past checkpoint expeiment from where to continue training')
parser.add_argument('-ld', '--log_dir', type=str, default='../Output/Log', help='Summaries log directory')
parser.add_argument('-cd', '--checkpoint_dir', type=str, default='../Output/Checkpoint', help='Checkpoint Saved')
parser.add_argument('-p', '--patch_size', type=int, default=32, help='Size of patch.')
parser.add_argument("--seed", type=int, default=547, help='Seed for tensorflow and numpy.')
parser.add_argument('-e', '--no_of_epoch', type=int, default=3, help='Number of epochs for training.')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
parser.add_argument('-a', '--lambda1', type=float, default=10, help='Paramter to balance generative loss.')
parser.add_argument('-b', '--lambda2', type=float, default=0.0001, help='Paramter to balance regularization loss.')

a = parser.parse_args()  
log_dir = os.path.join(a.log_dir, a.experiment_name + '_Fold_' + str(a.fold_number))
checkpoint_dir = os.path.join(a.checkpoint_dir, a.experiment_name + '_Fold_' + str(a.fold_number))

df_clinical = pd.read_csv(a.input_clinicaldata, sep='\t')
df = pd.read_csv(a.input_csv, sep=',')
_index_in_epoch = 0

#Functions
def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    i = 0
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))
    for i in range(1,num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)
    
def R2(tv,pv):
    res = np.sum(np.square(tv-pv))
    var = np.sum(np.square(tv-np.mean(tv)))
    return 1 - (res / var)
        
def train():
    #Save the results of each epochs
    lossT = np.zeros([3, a.no_of_epoch])
    lossG = np.zeros([3, a.no_of_epoch]) #Generative
    lossD = np.zeros([3, a.no_of_epoch]) #Discriminative
    lossR = np.zeros([3, a.no_of_epoch]) #Regularization
    results = {}
    labels = a.clinical_phenotype
    labels = labels.split(',')
    label_type = a.clinical_phenotype_datatype
    label_type = label_type.split(',')
    num_regression = 0
    regression_labels = []
    for i in range(len(labels)):
        l = labels[i]
        t = label_type[i]
        if 'float' in t:
            results['RSquare_'+l] = np.zeros([2, a.no_of_epoch])
            num_regression+= 1
            regression_labels.append(l)
        else:
            print("Error!! Unsupport label type, ", t)
            sys.exit()
    if 'FEV1pp_utah' in labels and 'FEV1_FVC_utah' in labels:
        results['precentageAccuracy_finalGold'] = np.zeros([2, a.no_of_epoch])
        results['precentageAccuracyOneOff_finalGold'] = np.zeros([2, a.no_of_epoch])

    confounder = a.confounder
    confounder = confounder.split(',')
    #Create the model
    # Input placeholders
    with tf.name_scope('input'):
        #Input subject x. Each subject is an array of N patches(32 x 32 x 32), 
        #where N varies with subject
        #Nb: Number of patches in a batch
        x = tf.placeholder(tf.float32, [None, a.patch_size, a.patch_size, a.patch_size], name='x-input') #[Nb, 32, 32, 32]
        #Clinical values to be predicted.
        #Ns: number of subjects
        y_ = tf.placeholder(tf.float32, [None , num_regression], name='y-input') #[Ns,?]]
        #confounders
        if len(confounder) > 0:
            c_ = tf.placeholder(tf.float32, [None , len(confounder)], name='con-founder-input') #[Ns, ?]
        isTrain = tf.placeholder(tf.bool) 
        #Number of patches in this batch.
        Nb = tf.placeholder(tf.int32)
        
    with tf.name_scope('input_x_reshape'):
        x_image = tf.reshape(x, [-1, a.patch_size, a.patch_size, a.patch_size, 1]) #[Nb, 32, 32, 32, 1]

    HiddenRepresentation = make_parallel(encoder, a.num_gpus, _inputs=x_image) #[Nb,1,1,1,128]

    X_Generated = make_parallel(decoder, a.num_gpus, _inputs=HiddenRepresentation) #[Nb,32,32,32,1] 
    X_Reconstructed = tf.reshape(X_Generated, [-1, a.patch_size, a.patch_size, a.patch_size]) #[Nb, 32, 32, 32]

    if len(confounder) > 0:
        y_predicted, attentionWeights, subjectFeature = discriminativeNetwork_confounder(HiddenRepresentation, isTrain, Nb, a.dropout, a.seed, num_regression, c_)
    else:
        y_predicted, attentionWeights, subjectFeature = discriminativeNetwork(HiddenRepresentation, isTrain, Nb, a.dropout, a.seed, num_regression)
                                                    
    
    with tf.name_scope('Loss'):
        loss_generative = tf.reduce_mean((x - X_Reconstructed)**2)  #L2 loss function        
        loss_discriminative = tf.reduce_sum(tf.reduce_mean((y_ - y_predicted)**2, axis=0))#L2 
        #Regularization
        epsilon = 0.0001
        alpha_epsilon = tf.add(attentionWeights, epsilon)
        log_alpha = tf.log(alpha_epsilon)
        loss_regularization = tf.reduce_sum(log_alpha)
        #Parameters
        print("lambda2: ", a.lambda2)
        print("lambda1: ", a.lambda1)
        loss = loss_discriminative + tf.scalar_mul(a.lambda1,loss_generative) + tf.scalar_mul(a.lambda2, loss_regularization)
    tf.summary.scalar('Loss', loss)
    
    with tf.name_scope('train'): 
        train_step = tf.train.AdamOptimizer(a.learning_rate).minimize(loss,
    colocate_gradients_with_ops=True)

    #Create a session
    sess = tf.InteractiveSession()
    # Add the model graph to TensorBoard
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    
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
    
    
    #Start executing the model
    for epoch in range(a.no_of_epoch):
        print("Epoch number: " + str(format(datetime.now()) ) + " " + str(epoch))
        total_loss = 0.0
        loss_g = 0.0
        loss_d = 0.0
        loss_r = 0.0
        TrueValue = []
        PredictedValue = []
        TrueGold = []
        i = 0
        while True:
            #xs: subject; ys:clinical values; gs:gold score; ns: subject name
            xs, ys, gs, ns, cs = next_batch(i)
            #pdb.set_trace()
            if np.sum(np.isnan(ys)) != 0:
                print(ns)
                print("Values are nan")
                continue
            if xs.shape[0] == 0:
                break
            if xs.shape[0]%a.num_gpus != 0:
                extra = a.num_gpus - xs.shape[0]%a.num_gpus
                for ii in range(extra):
                    temp = np.zeros([1,32,32,32])
                    xs = np.append(xs,temp,0)
                    
            [temp, _loss, _X_Reconstructed, _HiddenRepresentation, _y_predicted, _loss_generative, _loss_discriminative, \
            _attentionWeights, _loss_regularization] = sess.run([train_step, loss, X_Reconstructed, \
            HiddenRepresentation, y_predicted, loss_generative, loss_discriminative, attentionWeights, \
            loss_regularization], feed_dict={x:xs, isTrain:True, y_: ys, c_:cs, Nb:xs.shape[0]}) 
            if np.isnan(_loss) or np.isnan(_loss_generative) or np.isnan(_loss_discriminative) or np.isnan(_loss_regularization):
                print("Nan error loss")
                print(ns)
                continue

            total_loss += _loss
            loss_g += _loss_generative
            loss_d += _loss_discriminative
            loss_r += _loss_regularization
            TrueGold.append(gs)
            for v in ys:
                TrueValue.append(v)
            for v in _y_predicted:
                PredictedValue.append(v)
            i += 1
        total_loss /= i
        loss_g /= i
        loss_d /= i
        loss_r /= i
        
        lossT[0][epoch] =  total_loss
        lossG[0][epoch] = loss_g
        lossD[0][epoch] = loss_d
        lossR[0][epoch] = loss_r 
        TrueValue = np.asarray(TrueValue)
        PredictedValue = np.asarray(PredictedValue)
        TrueGold = np.asarray(TrueGold)
        
        FEV1_index = -1
        FEV1_FVC_index = -1
        for i in range(len(regression_labels)):
            l = regression_labels[i]
            results['RSquare_'+l][0][epoch] = R2(TrueValue[:,i],PredictedValue[:,i])
            if l == 'FEV1pp_utah':
                FEV1_index = i
            if l == 'FEV1_FVC_utah':
                FEV1_FVC_index = i
        if FEV1_index != -1 and FEV1_FVC_index != -1:
            #Train Gold score classifier
            data = {}
            fev1 = TrueValue[:,FEV1_index]
            fev1 = np.power(10, fev1)
            fev1_fvc = TrueValue[:,FEV1_FVC_index]
            fev1_fvc = np.power(10, fev1_fvc)
            data['FEV1pp_utah'] = fev1
            data['FEV1_FVC_utah'] = fev1_fvc
            data['finalGold'] = TrueGold
            yy, XX = dmatrices('finalGold' + ' ~ ' + 'FEV1pp_utah + FEV1_FVC_utah - 1', data=data, return_type='matrix')
            FEV2Gold = DecisionTreeClassifier()
            FEV2Gold.fit(XX,yy)
        
            gold_pred = FEV2Gold.predict( np.vstack([10**PredictedValue[:,FEV1_index],10**PredictedValue[:,FEV1_FVC_index]]).T ) 
            results['precentageAccuracy_finalGold'][0][epoch] = accuracy_score(TrueGold, gold_pred)
            #Gold accuracy one-off
            correct = 0
            for i in range(0, len(TrueGold)):
                if gold_pred[i] >= TrueGold[i]-1 and  gold_pred[i] <= TrueGold[i]+1:
                    correct += 1
        
            results['precentageAccuracyOneOff_finalGold'][0][epoch] = float(correct)/float(len(TrueGold))

        
        print("{} Training Loss = {:.4f}".format(datetime.now(), total_loss)) 
    
        #Run testing 
        if epoch%2 == 0 or epoch == a.no_of_epoch-1:
            print("Start testing " + str(format(datetime.now())))
            total_loss = 0.0
            loss_g = 0.0
            loss_d = 0.0
            loss_r = 0.0
            TrueValue = []
            PredictedValue = []
            TrueGold = []
            i = 0
            while True:
                #xs: subject; ys:clinical values; gs:gold score; ns: subject name
                xs, ys, gs, ns, cs = next_batch(i,3)
                if np.sum(np.isnan(ys)) != 0:
                    print(ns)
                    print("Values are nan in test")
                    continue
                if xs.shape[0] == 0:
                    break
                if xs.shape[0]%2 != 0:
                    temp = np.zeros([1,32,32,32])
                    xs = np.append(xs,temp,0)
                [_loss, _X_Reconstructed, _HiddenRepresentation, _y_predicted, _loss_generative, _loss_discriminative, _attentionWeights, _loss_regularization] = sess.run([loss, X_Reconstructed, HiddenRepresentation, y_predicted, loss_generative, loss_discriminative, attentionWeights, loss_regularization], feed_dict={x:xs, isTrain:False, y_: ys, c_:cs, Nb:xs.shape[0]})
                if np.isnan(_loss) or np.isnan(_loss_generative) or np.isnan(_loss_discriminative) or np.isnan(_loss_regularization):
                    print("Nan error loss in test")
                    print(ns)
                    continue
                total_loss += _loss
                loss_g += _loss_generative
                loss_d += _loss_discriminative
                loss_r += _loss_regularization
                TrueGold.append(gs)
                for v in ys:
                    TrueValue.append(v)
                for v in _y_predicted:
                    PredictedValue.append(v)
                i += 1
            total_loss /= i
            loss_g /= i
            loss_d /= i
            loss_r /= i
            lossT[2][epoch] =  total_loss
            lossG[2][epoch] = loss_g
            lossD[2][epoch] = loss_d
            lossR[2][epoch] = loss_r 
            TrueValue = np.asarray(TrueValue)
            PredictedValue = np.asarray(PredictedValue)
            TrueGold = np.asarray(TrueGold)
            
            FEV1_index = -1
            FEV1_FVC_index = -1
            for i in range(len(regression_labels)):
                l = regression_labels[i]
                results['RSquare_'+l][1][epoch] = R2(TrueValue[:,i],PredictedValue[:,i])
                if l == 'FEV1pp_utah':
                    FEV1_index = i
                if l == 'FEV1_FVC_utah':
                    FEV1_FVC_index = i
            if FEV1_index != -1 and FEV1_FVC_index != -1:
                gold_pred = FEV2Gold.predict( np.vstack([10**PredictedValue[:,FEV1_index],10**PredictedValue[:,FEV1_FVC_index]]).T ) 
                results['precentageAccuracy_finalGold'][1][epoch] = accuracy_score(TrueGold, gold_pred)
                #Gold accuracy one-off
                correct = 0
                for i in range(0, len(TrueGold)):
                    if gold_pred[i] >= TrueGold[i]-1 and  gold_pred[i] <= TrueGold[i]+1:
                        correct += 1
            
                results['precentageAccuracyOneOff_finalGold'][1][epoch] = float(correct)/float(len(TrueGold))

            print("{} Test Loss = {:.4f}".format(datetime.now(), total_loss))
        
        #save checkpoint of the model
        print("{} Saving checkpoint of model...".format(datetime.now())) 
        checkpoint_name = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name) 
        #Save the graph
        graph_def = tf.get_default_graph().as_graph_def()
        graph_txt = str(graph_def)
        with open(os.path.join(log_dir, 'graph.txt'), 'wt') as f: f.write(graph_txt)
        train_writer.close()
        np.save(os.path.join(log_dir, 'loss.npy'), lossT)
        np.save(os.path.join(log_dir, 'lossR.npy'), lossR)
        np.save(os.path.join(log_dir, 'lossD.npy'), lossD)
        np.save(os.path.join(log_dir, 'lossG.npy'), lossG)  
        df_result = pd.DataFrame()
        for key, val in results.items():
            df_result[key+'_Train'] = val[0]
            df_result[key+'_Test'] = val[1]
        df_result.to_csv(os.path.join(log_dir, 'results.csv'), index=None)

#Function to get data for processing        
def next_batch(step, train=1): 
#Step: defines of its the start of epoch. Train: 1: Train, 2: Validation 3: Test
    global _index_in_epoch
    global df_train
    global df_test
    
    start = _index_in_epoch   
    
    labels = a.clinical_phenotype
    labels = labels.split(',')
    label_type = a.clinical_phenotype_datatype
    label_type = label_type.split(',')
    
    #Training Samples
    if train == 1:
        num = df_train.shape[0]
        if step == 0:          
            # Start epoch   # Shuffle the data            
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            # Start next epoch
            start = 0
        counter = start
        while counter < num:
            try:
                #pdb.set_trace()
                record = df_train.iloc[counter]   
                currentSubject = np.load(os.path.join(a.data_dir, record['patch']))
                index = np.where(np.isnan(currentSubject))
                currentSubject[index] = -1024
                currentSubject = np.interp(currentSubject, [-1024,240], [0,1])
                con_founder = []
                confounder = a.confounder
                confounder = confounder.split(',')
                for c in confounder:
                    con_founder.append(record[c])
                regression = []
                for i in range(len(labels)):
                    l = labels[i]
                    t = label_type[i]
                    if t == 'float-log':
                        regression.append(np.log10(record[l]))
                    elif t == 'float':
                        regression.append(record[l])
                    else:
                        print("Error!! Unsupported label type ", t, l)
                        sys.exit()
                
                gold = record['finalGold']
                subjectName = record['sid']
                currentLabel = np.asarray(regression)
                con_founder = np.asarray(con_founder)
                counter += 1
                break
            except:
                print("Train", counter, num, record)
                counter += 1
                start += 1
                pdb.set_trace()
                continue

        if counter >= num: #no more elements can be fit into this batch
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        
        #Start the next epoch
        _index_in_epoch = counter
        currentLabel = np.reshape(currentLabel, [1,-1])
        con_founder = np.reshape(con_founder, [1,-1])
        return currentSubject, currentLabel, gold, subjectName, con_founder
        
    #Testing Samples
    else:
        num = df_test.shape[0]
        if step == 0:          
            # Start epoch   # Shuffle the data            
            df_test = df_test.sample(frac=1).reset_index(drop=True)
            # Start next epoch
            start = 0
        counter = start
        while counter < num:
            try:
                #pdb.set_trace()
                record = df_test.iloc[counter]   
                currentSubject = np.load(os.path.join(a.data_dir, record['patch']))
                index = np.where(np.isnan(currentSubject))
                currentSubject[index] = -1024
                currentSubject = np.interp(currentSubject, [-1024,240], [0,1])
                con_founder = []
                confounder = a.confounder
                confounder = confounder.split(',')
                for c in confounder:
                    con_founder.append(record[c])
                regression = []
                for i in range(len(labels)):
                    l = labels[i]
                    t = label_type[i]
                    if t == 'float-log':
                        regression.append(np.log10(record[l]))
                    elif t == 'float':
                        regression.append(record[l])
                    else:
                        print("Error!! Unsupported label type ", t, l)
                        sys.exit()
                
                gold = record['finalGold']
                subjectName = record['sid']
                currentLabel = np.asarray(regression)
                con_founder = np.asarray(con_founder)
                counter += 1
                break
            except:
                print("Train", counter, num, record)
                counter += 1
                start += 1
                continue

        if counter >= num: #no more elements can be fit into this batch
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        
        #Start the next epoch
        _index_in_epoch = counter
        currentLabel = np.reshape(currentLabel, [1,-1])
        con_founder = np.reshape(con_founder, [1,-1])
        return currentSubject, currentLabel, gold, subjectName, con_founder

def process_csv():
    global df
    global df_clinical
    global a
    
    if 'sid' not in df.columns:
        print("There is no 'sid' columns in input csv: %s" % (input_csv))
        print("Error!! The input CSV file %s should contain a column name: sid" % (a.input_csv))
        sys.exit()
    if 'patch' not in df.columns:
        print("Error!! The input CSV file %s should contain a column name: patch"% (a.input_csv))
        sys.exit()
    if 'fold'+str(a.fold_number) not in df.columns:
        print("Error!! The input CSV file %s should contain a column name: fold%d" %(a.input_csv, a.fold_number))
        sys.exit()

    # merge two dataframes
    df = df.merge(df_clinical, on='sid', how='left')

    # check the confounders
    confounder = a.confounder
    confounder = confounder.split(',')
    for c in confounder:
        if c not in df.columns:
            print("Error!! The input CSV file %s should contain a column name: %s"  %(a.input_clinicaldata, c))
            sys.exit()
        nan_rows = df[df[c].isnull()]
        if nan_rows.shape[0] > 0:
            df =  df[~ df[c].isnull()]
            print("Removing %d rows with nan in column: %s" % (nan_rows.shape[0], c))
        temp = np.unique(np.asarray(df[c]))
        if temp.shape[0] > 10: 
            print("Performing zscore on confounder: ", c)
            df[c] =  stats.zscore(np.asarray(df[c]), axis=0,ddof=1)
        if 'gender' in c.lower():
            df[c] = df.apply(lambda row: 0 if row[c] != 1 else 1 , axis=1)

    # Check the target labels to predict
    labels = a.clinical_phenotype
    labels = labels.split(',')
    label_type = a.clinical_phenotype_datatype
    label_type = label_type.split(',')
    if len(labels) != len(label_type):
        print("Error!! The number of target phenotypes and their types should match.")
        sys.exit()

    for i in range(len(labels)):
        l = labels[i]
        t = label_type[i]
        if l not in df.columns:
            print("Error!! The input CSV file %s should contain a column name %s " % (a.input_clinicaldata, l))
            sys.exit()
        try:
            df[l] = pd.to_numeric(df[l])
        except:
            print("Error!! The clinical phenotype %s to use as labels in input CSV file should be numerical values like float or int" % (l))
            sys.exit()
        nan_rows = df[df[l].isnull()]
        if nan_rows.shape[0] > 0:
            df =  df[~ df[l].isnull()]
            print("Removing %d rows with nan in column: %s" % (nan_rows.shape[0], l))

        if t not in ['float','float-log']:
            print("Error!! unsupported datatype: ", t , " for clinical phenotype: ", l)
            sys.exit()
    
    # Divide into train-val-test
    df_train = df.loc[df['fold'+str(a.fold_number)] == 'Train']
    df_test = df.loc[df['fold'+str(a.fold_number)] == 'Test']            

    return df_train, df_test
        
def main():
    global df_train
    global df_test
    global log_dir
    global checkpoint_dir
    
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)

    while tf.gfile.Exists(log_dir):
        print("Log Directory Already Exists: " + log_dir)
        log_dir = log_dir + "_1"
    tf.gfile.MakeDirs(log_dir)
    while tf.gfile.Exists(checkpoint_dir):
        print("Checkpoint Directory Already Exists: " + checkpoint_dir)
        checkpoint_dir = checkpoint_dir + '_1'
    tf.gfile.MakeDirs(checkpoint_dir)
    
    df_train, df_test = process_csv()
    print("Number of training subjects: ", df_train.shape[0])  
    print("Number of testing subjects: ", df_test.shape[0])
        
    train()


main()
