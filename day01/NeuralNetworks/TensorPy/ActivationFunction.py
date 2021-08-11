from typing import Any
import numpy as np
from ComputeGraph import Operation
class sigmoid(Operation):
    """
    返回元素x的sigmoid结果
    """
    def __init__(self, a) -> None:
        super().__init__(input_nodes=[a])
    
    def compute(self, a_value: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-a_value))

class softmax(Operation):
    def __init__(self, input_nodes: Any) -> None:
        super().__init__(input_nodes=[input_nodes])
    
    def compute(self, a_value) -> np.ndarray:
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]
        
    
    
    
