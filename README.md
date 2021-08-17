# Learning-Gan
 my notebook for learning gan network

- [Learning-Gan](#learning-gan)
  - [Implement the deep learning framework TensorPy by myself](#implement-the-deep-learning-framework-tensorpy-by-myself)

## Implement the deep learning framework TensorPy by myself

1. computing graph

```mermaid
graph LR
id1((a))
id2((b))
id3((a x b))
id4((ab + c))
id5((c))

id1 --> id3 --> id4;
id2 --> id3;
id5 --> id4;
```
$$
{
\left[ \begin{array}{cc}
2 & 1 \\
-1& -2\\
\end{array}
\right ]}
\times
{
    \left[\begin{array}{cc}
    1 \\
    1 \\    
    \end{array}
    \right]
}
 +
{\left[\begin{array}{cc}
    3\\
    3\\
\end{array}\right]
}=
{
    \left[\begin{array}{cc}
    6 \\
    0 \\    
    \end{array}
    \right]
}
$$

