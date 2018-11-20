import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

'''
Some Notes:

1. Fixed Batch Size 
Batch size is fixed instead of the usual 'None'. This is because in Natural Gradients,
we want to calculate a jacobian and Tensorflow doesn't have a nice way to do this. The
only way I found was to do a for loop, but we can't for loop over 'None'. Hence,
batch size is fixed. Is there a better way to do this?

2. Comparisons
We compare between AdamOptimizer and Natural Gradients using the vanilla GradientDescentOptimizer.
We fix the learning rate to be the same in both cases. Is this a fair comparison?

3. CG Errors
In our tests, the CG implementation seems to work well for very small dimensions but the errors seem
to build up as dimensions increase. In fact, we can't even assert using np.allclose without changing
the rtol (relative tolerance). Why?
'''

########################
##### SIMPLE MNIST #####
########################

# Hyperparams
learning_rate = 0.01
n_itrs = 10
batch_size = 100
hidden_units = 50

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

h1 = tf.layers.dense(X, hidden_units, activation=tf.nn.relu)
h2 = tf.layers.dense(h1, 10)

y_hat = tf.nn.softmax(h2)

all_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h2)
loss = tf.reduce_mean(all_loss)
output_class = tf.argmax(y_hat,1)
num_correct = tf.cast(tf.equal(tf.argmax(y_hat,1), tf.argmax(Y,1)), tf.float32)
accuracy = tf.reduce_mean(num_correct)

######################
##### OPTIMIZERS #####
######################

def run_adam():
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(n_itrs):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_X, Y: batch_Y}
            _, _loss, _accuracy = sess.run([train_step, loss, accuracy], feed_dict=feed_dict)

            print('Iteration: {}, training loss = {}, training accuracy = {}'.format(i, _loss, _accuracy))
            val_batch_X, val_batch_Y = mnist.test.next_batch(batch_size)
            val_feed_dict = {X: val_batch_X, Y: val_batch_Y}
            _val_loss, _val_accuracy = sess.run([loss, accuracy], feed_dict=val_feed_dict)
            print('Iteration: {}, val loss = {}, val accuracy = {}'.format(i, _val_loss, _val_accuracy))

            all_train_loss.append(_loss)
            all_train_acc.append(_accuracy)
            all_val_loss.append(_val_loss)
            all_val_acc.append(_val_accuracy)
    return all_train_loss, all_train_acc, all_val_loss, all_val_acc

def run_ng():
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []

    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grads = [tf.gradients(loss, param)[0] for param in trainable_variables]
    flattened_grads = flatten_tensor_variables(grads)
    all_grads = [tf.gradients(one_loss, trainable_variables) for one_loss in tqdm(tf.unstack(all_loss, axis=0))]
    flattened_all_grads = tf.stack([flatten_tensor_variables(one_grad) for one_grad in all_grads], axis=0)
    flattened_all_grads -= tf.reduce_mean(flattened_all_grads, axis=0, keepdims=True) # is this necessary?
    fish = tf.matmul(flattened_all_grads, flattened_all_grads, transpose_a=True) / batch_size
    ngs_ph = tf.placeholder(tf.float32, (flattened_all_grads.shape[1]))
    reshaped_ng = unflatten_tensor_variables(ngs_ph, [param.get_shape() for param in trainable_variables])
    l = list(zip(reshaped_ng, trainable_variables))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.apply_gradients(l)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(n_itrs):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_X, Y: batch_Y}
            _flattened_grads, _fish = sess.run([flattened_grads, fish], feed_dict=feed_dict)
            ngs = cg(_fish, _flattened_grads)
            feed_dict[ngs_ph] = ngs
            _loss, _accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)
            sess.run(train_step)

            print('Iteration: {}, training loss = {}, training accuracy = {}'.format(i, _loss, _accuracy))
            val_batch_X, val_batch_Y = mnist.test.next_batch(batch_size)
            val_feed_dict = {X: val_batch_X, Y: val_batch_Y}
            _val_loss, _val_accuracy = sess.run([loss, accuracy], feed_dict=val_feed_dict)
            print('Iteration: {}, val loss = {}, val accuracy = {}'.format(i, _val_loss, _val_accuracy))

            all_train_loss.append(_loss)
            all_train_acc.append(_accuracy)
            all_val_loss.append(_val_loss)
            all_val_acc.append(_val_accuracy)
    return all_train_loss, all_train_acc, all_val_loss, all_val_acc

def plot(adam_results, ng_results):
    titles = ['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']
    y_labels = ['Loss', 'Accuracy', 'Loss', 'Accuracy']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(list(range(len(adam_results[i]))), adam_results[i], label='Adam')
        plt.plot(list(range(len(ng_results[i]))), ng_results[i], label='NG')
        plt.title(titles[i])
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel(y_labels[i])
    plt.savefig('results.jpg')

############################
##### HELPER FUNCTIONS #####
############################

def flatten_tensor_variables(ts):
    return tf.concat(axis=0, values=[tf.reshape(x, [-1]) for x in ts])

def unflatten_tensor_variables(flatarr, shapes):
    arrs = []
    n = 0
    for shape in shapes:
        size = np.prod(list(shape))
        arr = tf.reshape(flatarr[n:n + size], shape)
        arrs.append(arr)
        n += size
    return arrs

##### Useful for doing pinv in Tensorflow
# def pinv(A, b, reltol=1e-6):
#     s, u, v = tf.svd(A)
#     atol = tf.reduce_max(s) * reltol
#     s = tf.boolean_mask(s, s > atol)
#     s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))
#     return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

def cg(A, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = np.matmul(A, p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

##################################
##### TEST CG IMPLEMENTATION #####
##################################

TEST_CG = False
def test_cg(m):
    '''
    Why does rtol need to be so big (1e-2) to pass the test?
    '''
    print('Testing CG for dimension {}'.format(m))
    with tf.variable_scope('test_cg_{}'.format(m)):
        A = tf.get_variable(name='A', trainable=False, shape=(m,m))
        sym_A = tf.matmul(A,A, transpose_b=True)
        b = tf.get_variable(name='b', trainable=False, shape=(m,))
        init = tf.global_variables_initializer()
    with tf.Session() as sess:
        for i in range(5):
            sess.run(init)
            _sym_A, _b = sess.run([sym_A, b])
            exact_result = np.matmul(np.linalg.inv(_sym_A), _b)
            approx_result = cg(_sym_A, _b, cg_iters=m*m*m)
            # print('Exact:', exact_result)
            # print('CG:', approx_result)
            assert np.allclose(exact_result, approx_result, rtol=1e-2)

if __name__ == '__main__':
    if TEST_CG:
        for i in range(2,20):
            test_cg(i)
    adam_results = run_adam()
    ng_results = run_ng()
    plot(adam_results, ng_results)
    
