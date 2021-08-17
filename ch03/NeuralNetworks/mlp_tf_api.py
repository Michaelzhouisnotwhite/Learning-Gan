"""
@author: Michael
@version: 2021-08-14
"""
# %%
from abc import ABC

import tensorflow as tf
import tensorflow.keras as ks
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
X = np.concatenate((blue_points, red_points))
Y = np.array([[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points))


# %%
class MLP(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()
        # Flatten层将除第一维（batch_size）以外的维度展平
        # self.flatten = tf.keras.layers.Flatten()
        # 全连接层
        self.dense1 = tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid, input_shape=(1, 2))
        self.dense2 = tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)

    def __call__(self, inputs):  # [batch_size, 28, 28, 1]
        # x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(inputs)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        return x


# model = ks.models.Sequential([
#     ks.layers.Dense(units=2, activation=tf.nn.sigmoid, input_shape=(1, 2)),
#     tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
# ])
num_epochs = 1000  # 训练轮数
learning_rate = 0.01  # 学习率
model = MLP()  # 实例化模型
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for batch_index in range(num_epochs):
    # 随机取一批训练数据
    with tf.GradientTape() as tape:
        # 计算模型预测值
        y_pred = model(X)
        # 计算损失函数
        loss = tf.keras.losses.mean_squared_logarithmic_error(y_true=Y, y_pred=y_pred)
        # loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    # 计算模型变量的导数
    grads = tape.gradient(loss, model.variables)
    # 优化器更新模型参数以减小损失函数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# %%
# model.summary()
# # %%
# model.compile(
#     optimizer='sgd',
#     loss=ks.losses.mean_squared_logarithmic_error,
#     metrics=['acc']
# )
# model.fit(X, Y, epochs=10000)
