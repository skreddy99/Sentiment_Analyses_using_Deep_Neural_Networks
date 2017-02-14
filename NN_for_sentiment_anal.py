'''

Sentiment analyses of pos.txt and neg.txt file with
a collection of positive comments and negative comments

'''
from Sentiment_Analyses import create_feature_sets_and_labels
import tensorflow as tf
import pickle
import numpy as np

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')


n_h1_nodes = 1500
n_h2_nodes = 1500
n_h3_nodes = 1500

n_classes = 2
batch_size = 1000
n_epochs = 30
print('Batch size is ', batch_size)
print('Num of epochs is ', n_epochs)
print('Num of L1 nodes is ', n_h1_nodes)
print('Learning rate is 0.001')

x = tf.placeholder('float')
y = tf.placeholder('float')

wb_layer1 = {'f_fum':n_h1_nodes,
             'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_h1_nodes])),
             'bias':tf.Variable(tf.random_normal([n_h1_nodes]))}

wb_layer2 = {'f_fum':n_h2_nodes,
             'weight':tf.Variable(tf.random_normal([n_h1_nodes, n_h2_nodes])),
             'bias':tf.Variable(tf.random_normal([n_h2_nodes]))}

wb_layer3 = {'f_fum':n_h3_nodes,
             'weight':tf.Variable(tf.random_normal([n_h2_nodes, n_h3_nodes])),
             'bias':tf.Variable(tf.random_normal([n_h3_nodes]))}

wb_layero = {'f_fum':None,
             'weight':tf.Variable(tf.random_normal([n_h3_nodes, n_classes])),
             'bias':tf.Variable(tf.random_normal([n_classes]))}




def nn_senti_analyses(data):
    h_layer1 = tf.add(tf.matmul(data, wb_layer1['weight']), wb_layer1['bias'])
    h_layer1 = tf.nn.relu(h_layer1)

    h_layer2 = tf.add(tf.matmul(h_layer1, wb_layer2['weight']), wb_layer2['bias'])
    h_layer2 = tf.nn.relu(h_layer2)

    h_layer3 = tf.add(tf.matmul(h_layer2, wb_layer3['weight']), wb_layer3['bias'])
    h_layer3 = tf.nn.relu(h_layer3)

    output = tf.add(tf.matmul(h_layer3, wb_layero['weight']), wb_layero['bias'])

    return output

def train_nn_senti(x):
    anal = nn_senti_analyses(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(anal, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss = 0
            i=0
            while i<len(train_x):
                start = i
                end = start+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _,c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                epoch_loss += c
                i+=batch_size

            print('Epoch' , epoch+1, 'completed out of ', n_epochs, 'loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(anal, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy: ', accuracy.eval({x:test_x, y:test_y}))

train_nn_senti(x)


