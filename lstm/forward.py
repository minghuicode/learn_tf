#coding:utf-8
import tensorflow as tf

HIDDEN_SIZE = 30
NUM_LAYERS = 2

def forward(X,Y):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)
    ])
    outputs, _ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    output = outputs[:,-1,:]
    
    predictions = tf.contrib.layers.fully_connected(
        output,1,activation_fn=None
    )
    return predictions
