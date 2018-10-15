import tensorflow as tf
from sklearn import datasets
import numpy as np
def int_to_vector(data,label_size):
    vector = np.zeros((len(data),label_size),dtype='f')
    for i,single in enumerate(data):
        vector[i][single]=1
    return vector
iris = datasets.load_iris()
input =iris['data']
output = iris['target']
feature_size = 4
second_layer_size = 4
label_size = 3
output=int_to_vector(output,label_size)
tf.set_random_seed(1)
train_inputs = tf.placeholder(tf.float32, shape=[None,feature_size])
train_labels = tf.placeholder(tf.float32, shape=[None,label_size])
w1 = tf.Variable(tf.random_uniform([feature_size, second_layer_size], -1.0, 1.0))
w2 = tf.Variable(tf.random_uniform([second_layer_size, label_size], -1.0, 1.0))
pass_1 = tf.nn.sigmoid(tf.matmul(train_inputs,w1))
vali=tf.matmul(pass_1,w2)
pass_2 = tf.nn.softmax(tf.matmul(pass_1,w2))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pass_2, labels=train_labels))
#loss = tf.reduce_mean(tf.abs(train_labels-pass_2))
session=tf.Session()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
init = tf.global_variables_initializer()
session.run(init)

epoch = 30000
feed_dict = {train_inputs: input, train_labels: output}
for i in range(0,epoch):
    pred,mid,intt,_, cur_loss = session.run([pass_2,pass_1,train_inputs,optimizer, loss], feed_dict=feed_dict)
    if i%1000==0:
        print(cur_loss)

print([np.argmax(x) for x in pred])