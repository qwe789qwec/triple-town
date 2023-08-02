import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# plt.switch_backend('Agg')

class Neuron:
    def __init__(self, inputs, outputs, learning_rate, activation):
        self.weights = np.random.randn(inputs, outputs) + 1
        self.bias = np.random.randn(1, outputs) + 1
        # self.weights = np.zeros((inputs, outputs))
        # self.bias = np.zeros((1, outputs))
        # self.weights = np.random.randn(size)
        self.activation = activation
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        self.aoutput = self.activation(self.output)

    def backward(self, delta):
        # error = self.aoutput - target
        # 計算權重的梯度
        # weights_gradient = np.sum(error * self.input)
        # 計算偏差值的梯度
        self.weights -= np.dot(self.input.T, delta) * self.learning_rate
        # self.weights -= self.input.T * delta * self.learning_rate
        self.bias -= delta * self.learning_rate

class activation:
    def equal(self, x):
        return x
    def deequal(self, x):
        return 1
    def step(self, x):
        if(x > 0):
            return 1
        else:
            return 0.001
    def ReLU(self, x):
        return np.maximum(0.001, x)
    def deReLU(self, x):
        return np.where(x > 0, 1, 0.01)
    def sigmoid(self, x):
        output = 1 / (1 + np.exp(-x))
        return np.clip(output, 1e-7, 1-1e-7)
    def desigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis = 1, keepdims = True))
        probabilities = exp / np.sum(exp, axis = 1, keepdims = True)
        return probabilities

def makegif(x, y1, y2, number):
    plt.plot(x, y2, 'o', label='sin(x)', color = "b")
    plt.plot(x, y1, 'o', label='predicted', color = "r")
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close()
    plt.title(f'Plot {number+1}')
    plt.savefig(f'plot_{number+1}.png')
    plt.clf()

def printparameter(parameter):
    print("[", end="")
    for number in parameter.flatten():
        # print(round(number, 2),", ", end="")
        print(number,", ", end="")
    print("]")

Neuron_size = 24
max_epoch = 1000
learning_threshold = 10
learning_rate = 0.001
act_function = activation()
first = Neuron(1, Neuron_size, learning_rate, act_function.sigmoid)
second = Neuron(Neuron_size, Neuron_size, learning_rate, act_function.sigmoid)
third = Neuron(Neuron_size, 1, learning_rate, act_function.sigmoid)
sample_point = 200
x = np.random.uniform(-2 * np.pi, 2 * np.pi, sample_point)
# x = np.arange(0, 2 * np.pi, 0.01)

batch_size = 10  # 每次提取的数字数量
num_batches = len(x) // batch_size  # 计算可以提取多少批次

images = []

# first.weights = np.array([-2.78])
# first.bias = np.array([9.30])
# second.weights = np.array([1.06])
# second.bias = np.array([-0.47])
# third.weights = np.array([7.07])
# third.bias = np.array([-2.75])
last_error = 999
error_count = 0

for episode in range(max_epoch):
    output = []
    error = []
    for i in range(len(x)):
    #     start_idx = i * batch_size
    #     end_idx = start_idx + batch_size
    #     batch = x[start_idx:end_idx]
        # print(i)
        data = x[i]

        first.forward(data)
        second.forward(first.aoutput)
        third.forward(second.aoutput)
        
        third_error = third.output - np.sin(x[i])
        third_delta = third_error
        second_error = np.dot(third_delta, third.weights.T)
        second_delta = second_error * act_function.desigmoid(second.aoutput)
        first_error = np.dot(second_delta, second.weights)
        first_delta = first_error * act_function.desigmoid(first.aoutput)
        third.backward(third_delta)
        second.backward(second_delta)
        first.backward(first_delta)

    # for i in range(len(x)):
    #     data = x[i]

    #     first.forward(data)
    #     second.forward(first.aoutput)
    #     third.forward(second.aoutput)
    #     output.append(third.output)
    #     third_error = third.output - np.sin(x[i])
    #     error.append(third_error)

    # if episode%100 == 0 or np.abs(np.sum(error)) < 50:
    #     makegif(x, np.reshape(output, len(x)), np.sin(x), episode)
    #     images.append(Image.open(f'plot_{episode+1}.png'))
    
    # if((last_error - np.abs(np.sum(error))) < learning_rate):
    #     error_count += 1
    #     if error_count > 5 and last_error < 100:
    #         learning_rate = learning_rate / 1.03
    #         error_count = 0
    #     else:
    #         learning_rate = learning_rate * 1.03
        
        # print(learning_rate)
    # print("error:", np.sum(error))
    # last_error = np.abs(np.sum(error))

# images[0].save('animated_plot.gif', save_all=True, append_images=images[1:], duration=200, loop=0)
# for i in range(int(max_epoch/100)):
#     os.remove(f'plot_{i*100+1}.png')

print("first.weights")
printparameter(first.weights)
print("first.bias")
printparameter(first.bias)
print("second.weights")
printparameter(second.weights)
print("second.bias")
printparameter(second.bias)
print("third.weights")
printparameter(third.weights)
print("third.bias")
printparameter(third.bias)

# error: [[-0.00441285]]
# first.weights [[-1.24821408]]
# first.bias [[3.74440627]]
# second.weights [[3.78927479]]
# second.bias [[0.15243486]]
# third.weights [[0.0088354]]
# third.bias [[0.13859234]]

output = []
for i in range(len(x)):
    data = x[i]

    first.forward(data)
    second.forward(first.aoutput)
    third.forward(second.aoutput)
    # print(third.output)
    output.append(third.output)
    third_error = third.output - np.sin(x[i])
    error.append(third_error)
    if i >= len(x) - 1:
        plt.plot(x, np.sin(x), 'o', label='sin(x)', color = "b")
        plt.plot(x, np.reshape(output, len(x)), 'o', label='predicted', color = "r")
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()
        plt.clf()

print("error:", np.sum(error))