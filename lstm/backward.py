#coding:utf-8
import tensorflow as tf 
import generateds
import forward
import os
import numpy as np
import matplotlib.pyplot as plt

TRAINING_STEPS = 10000
BATCH_SIZE = 32
MODEL_SAVE_PATH = './model/lstm/'
MODEL_NAME = 'lstm_model'

def backward(sess,train_X,train_Y):
        ds = tf.data.Dataset.from_tensor_slices((train_X,train_Y))
        ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
        x,y = ds.make_one_shot_iterator().get_next()

        with tf.variable_scope('model'):
            predictions = forward.forward(x,y)
            loss = tf.losses.mean_squared_error(labels=y,predictions=predictions)
            train_op = tf.contrib.layers.optimize_loss(
                loss,tf.train.get_global_step(),
                optimizer='Adagrad',                        
                learning_rate=0.1                        
                )
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for i in range(TRAINING_STEPS):
                _,l = sess.run([train_op,loss])
                if i % 1000 == 0:
                    print('train step: '+str(i)+', loss: '+str(l))
                    saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=i)

def run_eval(sess,test_X,test_Y):
    ds = tf.data.Dataset.from_tensor_slices((test_X,test_Y))
    ds = ds.batch(1)
    x,y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope("model",reuse=True):
        prediction = forward.forward(x,[.0])
        predictions = []
        labels = []
        for i in range(generateds.TEST_EXAMPLES):
            p, l = sess.run([prediction,y])
            predictions.append(p)
            labels.append(l)
        # calculate rmse
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels).squeeze()
        rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
        print('mean square error is: %.2f' % rmse)
        # plot predict sinx figure
        plt.figure()
        plt.plot(predictions,label='predictions')
        plt.plot(labels,label='real_sin')
        plt.legend()
        plt.show()  

def main():
    train_X,train_Y = generateds.generateds_train()
    test_X,test_Y = generateds.generateds_test()
    with tf.Session() as sess:
        backward(sess,train_X,train_Y)
        run_eval(sess,test_X,test_Y)

if __name__ == "__main__":
    main()
                
