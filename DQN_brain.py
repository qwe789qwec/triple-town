import numpy as np

learning_rate = 0.01
weight = 0.5
bias = 0.2

class DQN_brain:
    def neuron(input_features, weights, activation_func):
        # 線性組合
        linear_combination = np.dot(input_features, weights)

        # 非線性轉換
        output = activation_func(linear_combination)

        return output

    def relu(x):
        return np.maximum(0, x)

# 定義輸入和目標輸出
input_data = 0.8
target_output = 1

# 計算神經元的輸出
neuron_output = input_data * weight + bias

# 計算梯度
weight_gradient = -input_data * (target_output - neuron_output)
bias_gradient = -(target_output - neuron_output)

# 更新權重和偏差
weight -= learning_rate * weight_gradient
bias -= learning_rate * bias_gradient
