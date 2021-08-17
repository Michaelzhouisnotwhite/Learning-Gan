# %%
import numpy as np
import tensorflow as tf
import pylab as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
v1 = tf.compat.v1
v1.disable_v2_behavior()
# %%
# randn: 返回服从正态分布的随机值
# 生成一些集中在（-2, -2）的点
red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))
# 生成一些集中在（2， 2）的点
blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))
# %%
# v1.Graph().as_default()
# %%
X = v1.placeholder(dtype=tf.float64)
# %%
c = v1.placeholder(dtype=tf.float64)
# %%
# 权重矩阵
W = v1.Variable(np.random.randn(2, 2))
# %%
# 偏置
b = v1.Variable(np.random.randn(2))
# %%
# p = tf.sigmoid(tf.add(tf.matmul(X, W), b))
p = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
# %%
J = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(c, tf.math.log(p)), axis=1)))

# %%
minimization_op = v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

# %%
feed_dict = {X: np.concatenate((blue_points, red_points)), c: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)}
# %%
sess = v1.Session()
sess.run(v1.global_variables_initializer())
# %%
for step in range(100):
    J_value = sess.run(J, feed_dict=feed_dict)
    if step % 10 == 0:
        print("Step: [%s], Loss: [%s]" % (step, J_value))

    sess.run(minimization_op, feed_dict=feed_dict)
# %%
W_value = sess.run(W)
print("Weight Matrix:\n", W_value)
# %%
b_value = sess.run(b)
print("Bias:\n", b_value)

# %%

plt.scatter(red_points[:, 0], red_points[:, 1], color="red")
plt.scatter(blue_points[:, 0], blue_points[:, 1], color="blue")

x_axis = np.linspace(-4, 4, 100)
y_axis = -W_value[0][0] / W_value[1][0] * x_axis - b_value[0] / W_value[1][0]
plt.plot(x_axis, y_axis)
plt.show()

# %%
