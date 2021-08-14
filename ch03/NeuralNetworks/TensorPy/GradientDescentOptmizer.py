from queue import Queue
from typing import Any
from ComputeGraph import Operation, Variable
import numpy as np


class negative(Operation):
    def __init__(self, x: Any) -> None:
        """
        各个元素的负数

        Args:
            x (Any): 矩阵
        """
        super().__init__(input_nodes=[x])

    def compute(self, x_value) -> np.ndarray:
        return -x_value


class reduce_sum(Operation):
    def __init__(self, A: Any, axis: int = None) -> None:
        """
        矩阵中沿着某维度的总和

        Args:
            A (Any): 矩阵
            axis (int, optional): 某维度. Defaults to None.
        """
        super().__init__(input_nodes=[A])
        self.axis = axis

    def compute(self, A_value):
        return np.sum(A_value, self.axis)


class multiply(Operation):
    def __init__(self, x, y) -> None:
        """
        元素的点积运算

        Args:
            x: 矩阵
            y: 矩阵
        """
        super().__init__(input_nodes=[x, y])

    def compute(self, x_value, y_value):
        return x_value * y_value


class log(Operation):
    def __init__(self, x) -> None:
        super().__init__(input_nodes=[x])

    def compute(self, x_value):
        return np.log(x_value)


_gradient_registry = {}


class RegisterGradient:
    def __init__(self, op_type) -> None:
        self._op_type = eval(op_type)

    def __call__(self, f) -> Any:
        _gradient_registry[self._op_type] = f
        return f


@RegisterGradient('negative')
def _negative_gradient(op, grad):
    return -grad


def compute_gradients(loss):
    grad_table = {loss: 1}

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)
    while not queue.empty():
        node = queue.get()
        if node != loss:
            grad_table[node] = 0
            for consumer in node.consumers:
                loss_grad_wrt_consumer_output = grad_table[consumer]
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]
                loss_grad_wrt_consumer_inputs = bprop(consumer, loss_grad_wrt_consumer_output)
                if len(consumer.input_nodes) == 1:
                    grad_table[node] += loss_grad_wrt_consumer_inputs
                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    loss_grad_wrt_node = loss_grad_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += loss_grad_wrt_node
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)
    return grad_table


class GradientDescentOptimizer:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)
                for node in grad_table:
                    if isinstance(node, Variable):
                        grad = grad_table[node]
                        node.value -= learning_rate * grad

        return MinimizationOperation()
