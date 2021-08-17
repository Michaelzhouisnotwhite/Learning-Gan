# %%
import tensorflow as tf
import numpy as np
import pylab as plt
v1 = tf.compat.v1
v1.disable_v2_behavior()

# %%
red_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 0]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 1]] * 25)
))
blue_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 1]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 0]] * 25)
))

# %%
W_hidden = tf.Variable(np.random.randn(2, 2))
b_hidden = tf.Variable(np.random.randn(2))

# %%
W_output = tf.Variable(np.random.randn(2, 2))
b_output = tf.Variable(np.random.randn(2))
# %%
X = v1.placeholder(dtype=tf.float64)
c = v1.placeholder(dtype=tf.float64)

# %%
p_hidden = tf.sigmoid(tf.add(tf.matmul(X, W_hidden), b_hidden))
p_output = tf.nn.softmax(tf.add(tf.matmul(p_hidden, W_output), b_output))

# %%
J = tf.negative(tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.multiply(c, tf.math.log(p_output)), axis=1)))
# %%
minimization_op = v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)
# %%
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
}
# %%
sess = v1.Session()
# %%
sess.run(v1.global_variables_initializer())
# %%
for step in range(10000):
    J_value = sess.run(J, feed_dict)
    if step % 100 == 0:
        print("step: [%s], loss: [%s]" % (step, J_value))

    sess.run(minimization_op, feed_dict)


