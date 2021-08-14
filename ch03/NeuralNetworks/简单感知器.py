# %%
import numpy as np
import TensorPy as tp
import pylab as plt

# %%
# randn: 返回服从正态分布的随机值
# 生成一些集中在（-2, -2）的点
red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))
# 生成一些集中在（2， 2）的点
blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))
# %%
tp.Graph().as_default()
# %%
X = tp.placeholder()
# %%
c = tp.placeholder()
# %%
# 权重矩阵
W = tp.Variable(np.random.randn(2, 2))
# %%
# 偏置
b = tp.Variable(np.random.randn(2))
# %%
# p = tp.sigmoid(tp.add(tp.matmul(X, W), b))
p = tp.softmax(tp.add(tp.matmul(X, W), b))
# %%
J = tp.negative(tp.reduce_sum(tp.reduce_sum(tp.multiply(c, tp.log(p)), axis=1)))

# %%
minimization_op = tp.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

# %%
feed_dict = {X: np.concatenate((blue_points, red_points)), c: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)}
# %%
sess = tp.Session()
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
