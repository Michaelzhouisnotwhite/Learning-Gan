from abc import abstractmethod
from queue import Queue
# from GradientDescentOptmizer import RegisterGradient
from typing import Any, List, Union, Optional
import numpy as np


class Graph:
    """计算图
    """

    def __init__(self) -> None:
        self.operations = []  # 操作节点
        self.placeholders = []  # 占位符节点
        self.variables = []  # 变量节点

    def as_default(self):
        """默认计算出图
        """
        global _default_graph
        _default_graph = self


class Operation:
    """
    操作
    """

    def __init__(self, input_nodes: List['Operation'] = None) -> None:
        if input_nodes is None:
            input_nodes = []
        self.output = []
        self.inputs = []
        self.input_nodes = input_nodes
        self.consumers = []
        for input_node in input_nodes:
            input_node.consumers.append(self)
        _default_graph.operations.append(self)

    @abstractmethod
    def compute(self, *args):
        pass


class placeholder:
    """
    占位变量
    """

    def __init__(self) -> None:
        self.consumers = []
        self.output = []
        self.inputs = []
        _default_graph.placeholders.append(self)


class Variable:
    """
    变量
    """

    def __init__(self, initial_value: Union[list, np.ndarray]) -> None:
        self.value = initial_value
        self.consumers = []
        self.output = []
        self.inputs = []
        _default_graph.variables.append(self)


class matmul(Operation):
    """
    矩阵乘法
    """

    def __init__(self, x: Union[Variable, placeholder, Operation], y: Union[Variable, placeholder, Operation]) -> None:
        """__init__():

        Args:
            x (Union[Variable, placeholder, Operation]): x
            y (Union[Variable, placeholder, Operation]): y
        """
        super().__init__(input_nodes=[x, y])

    def compute(self, x_value: np.ndarray, y_value: np.ndarray) -> np.ndarray:
        """两个元素相乘

        Args:
            x_value (np.ndarray): x
            y_value (np.ndarray): y

        Returns:
            (np.ndarray)
        """
        return x_value.dot(y_value)


class add(Operation):
    """
    矩阵加法
    """

    def __init__(self, x: Union[Variable, placeholder, Operation], y: Union[Variable, placeholder, Operation]) -> None:
        super().__init__(input_nodes=[x, y])

    def compute(self, x_value: np.ndarray, y_value: np.ndarray):
        return x_value + y_value


def traverse_postorder(operation: Operation) -> list:
    """遍历计算图

    Args:
        operation (Operation): 节点

    Returns:
        list: 节点列表
    """
    nodes_postorder = []

    def recurse(node: Operation):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:
    @staticmethod
    def run(operation: Union[Operation, Variable, placeholder], feed_dict: dict = None) -> list:
        """计算操作输出

        Args:
            operation (Operation): 要计算输出的操作
            feed_dict (dict): placeholder提供的数据. Defaults to None.

        Returns:
            (list): 输出列表
        """
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if isinstance(node, placeholder):
                # 如果是placeholder类型，将对应类型的数据给他
                node.output = feed_dict[node]

            elif isinstance(node, Variable):
                # 变量本身就是输出
                node.output = node.value
            else:
                # 输入操作的节点
                node.inputs = [input_node.output for input_node in node.input_nodes]
                # compute()执行具体操作逻辑
                node.output = node.compute(*node.inputs)
            if isinstance(node.output, list):
                # 将list转换成ndarray类型
                node.output = np.array(node.output)

        return operation.output


if __name__ == "__main__":
    Graph().as_default()
    a = Variable([[2, 1], [-1, -2]])
    b = Variable([1, 1])
    c = placeholder()
    y = matmul(a, b)
    z = add(y, c)

    session = Session()
    output = session.run(y)
    print(output)
