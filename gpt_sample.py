import numpy as np
import matplotlib.pyplot as plt

# 创建输入数据集，在范围 [-2π, 2π] 内随机采样 1000 个点
x_train = np.random.uniform(0, 2 * np.pi, 1000).reshape(-1, 1)
y_train = np.sin(x_train)

# 数据归一化
x_mean = x_train.mean()
x_std = x_train.std()
x_train = (x_train - x_mean) / x_std

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std

# 定义神经网络模型
class NeuralNetwork:
    def __init__(self):
        np.random.seed(42)
        self.weights = np.random.randn(1, 64)
        self.bias = np.random.randn(1, 64)
        self.output_weights = np.random.randn(64, 1)
        self.output_bias = np.random.randn(1, 1)

    def forward(self, x):
        self.hidden_layer = np.dot(x, self.weights) + self.bias
        self.hidden_layer = np.maximum(0.01 * self.hidden_layer, self.hidden_layer)  # Leaky ReLU activation
        output = np.dot(self.hidden_layer, self.output_weights) + self.output_bias
        return output

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)  # Leaky ReLU gradient

    def backward(self, x, y, output, learning_rate=0.0001):
        output_error = output - y
        output_delta = output_error * 1  # identity activation gradient
        # print(output_delta.shape) #(1000, 1)
        hidden_error = np.dot(output_delta, self.output_weights.T)
        # print(hidden_error.shape) #(1000, 64)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_layer)  # Leaky ReLU gradient
        # print(hidden_delta.shape) #(1000, 64)

        self.output_weights -= learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.output_bias -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights -= learning_rate * np.dot(x.T, hidden_delta)
        self.bias -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# 创建模型实例
model = NeuralNetwork()

# 训练模型
for epoch in range(1000):
    output = model.forward(x_train)
    model.backward(x_train, y_train, output)

# 在 [-2π, 2π] 范围内生成用于预测的输入数据
x_test = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
x_test_normalized = (x_test - x_mean) / x_std
y_pred = model.forward(x_test_normalized) * y_std + y_mean  # 还原归一化后的预测结果

# 可视化预测结果
plt.plot(x_train * x_std + x_mean, y_train * y_std + y_mean, label='sin(x)')
plt.plot(x_test, y_pred, label='predicted')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()
