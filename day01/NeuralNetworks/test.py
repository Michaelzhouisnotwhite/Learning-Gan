# %%
import TensorPy as tp

# %%
tp.Graph().as_default()

# %%
a = tp.Variable([[2, 1],
                 [-1, -2]])
b = tp.Variable([1, 1])
c = tp.placeholder()


# %%
y = tp.matmul(a, b)

# %%
z  = tp.add(y, c)

# %%
sess = tp.Session()


# %%
output = sess.run(z, feed_dict={c: [3, 3]})

# %%
print(output)

# %%
