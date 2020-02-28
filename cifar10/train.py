""" 
Code for loading CIFAR10 and training was originally written 
for CS 231n at Stanford University (cs231n.stanford.edu). 
It has been modified for pruning.
For the original version, please visit cs231n.stanford.edu.  
"""
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import os
import shutil

from data import get_CIFAR10_data
from model import model
from prune_utils import get_prune_op

class Model:
    def __init__(self):
        # gpu option
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config)
    
    def load_cifar10(self):        
        # Invoke the above function to get our data.
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = get_CIFAR10_data()
        print('Train data shape: ', self.X_train.shape)
        print('Train labels shape: ', self.y_train.shape)
        print('Validation data shape: ', self.X_val.shape)
        print('Validation labels shape: ', self.y_val.shape)
        print('Test data shape: ', self.X_test.shape)
        print('Test labels shape: ', self.y_test.shape)
        
    def construct_model(self, ckpt_dir=None):
        # Define inputs
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.train.get_or_create_global_step()

        # ================================================================ #
        # YOUR CODE HERE:
        #   define our model
        #   save output of the model to self.y_out 
        # ================================================================ #
        
        self.y_out = model(self.X,self.y)
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        # Define our loss
        total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.y,10),logits=self.y_out)
        self.mean_loss = tf.reduce_mean(total_loss)

        # Define our optimizer
        self.optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
            
        # split train_step = optimizer.minimize(self.mean_loss)
        train_gradient = self.optimizer.compute_gradients(self.mean_loss)
        
        # initialize or load model parameters
        self.saver = tf.train.Saver(max_to_keep=10)
        if ckpt_dir is not None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_dir))
            print('Pre-trained model restored from %s' % (ckpt_dir))
        else:
            self.sess.run(tf.global_variables_initializer())
            print('Initialize variables')
        # ================================================================ #
        # YOUR CODE HERE:
        #   implement in prune_utils.py
        #   1.prune parameters based on your threshold 
        #     (make sure pruning is effectively applied in step 1)
        #   2.get pruned gradient update operator accordingly, save to prune_gradient
        # ================================================================ #
        
        prune_gradient = get_prune_op(self.sess, train_gradient, percentage=0.6)
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        # save pruned parameters
        tmp_dir = '__tmp__'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self.saver.save(self.sess, os.path.join(tmp_dir, 'pruned_model.ckpt'))       
        # define gradients and initialize optimizer parameters
        self.train_op = self.optimizer.apply_gradients(prune_gradient, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        # reload pruned parameters
        self.saver.restore(self.sess, tf.train.latest_checkpoint(tmp_dir))
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)        
        
        
    def run_model(self, sess, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              save_every=1, train_op=None, plot_losses=False):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(predict,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = train_op is not None
        
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [loss_val,correct_prediction,accuracy]
        if training_now:
            variables[-1] = train_op

        # counter 
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))): # TODO: ceil?? Is this right?!
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
                
                # create a feed dictionary for this batch
                feed_dict = {self.X: Xd[idx,:],
                             self.y: yd[idx],
                             self.is_training: training_now }
                # get batch size
                actual_batch_size = yd[idx].shape[0]
                
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = sess.run(variables,feed_dict=feed_dict)
                
                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
                
                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
            # Save the model checkpoint periodically.
            if e % save_every == 0:
                checkpoint_path = os.path.join('./ckpt/', 'model.ckpt')
                self.saver.save(sess, checkpoint_path, global_step=int(e+1))
        return total_loss,total_correct 
        
    def train(self):
        print('Training')
        self.run_model(self.sess,self.y_out,self.mean_loss,self.X_train,self.y_train,\
            epochs=10,batch_size=64,print_every=50,save_every=1,\
            train_op=self.train_op,plot_losses=False)
        print('Validation')
        self.run_model(self.sess,self.y_out,self.mean_loss,self.X_val,self.y_val,\
            epochs=1,batch_size=64)
        print('Testing')
        self.run_model(self.sess,self.y_out,self.mean_loss,self.X_test,self.y_test,\
            epochs=1,batch_size=64)
    


if __name__=="__main__":
    m = Model()
    m.load_cifar10()
    m.construct_model(ckpt_dir='ckpt')
    m.train()
