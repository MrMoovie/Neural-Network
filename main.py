import numpy as np

np.random.seed(0)  # check out the func

inputs = [[1.2, 2.3, -1.16]]


class LayerDense:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.1 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        n_calc = np.dot(inputs, self.weights) + self.biases

        activation = Activation_Softmax()  # / activation = Activation_Softmax()
        activation.forward(n_calc)

        self.outputs = activation.output


class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)


class Activation_Softmax:
    def forward(self, input):
        dec = input - np.max(input, axis=1, keepdims=True)
        exp_values = np.exp(dec)
        sum_values = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values / sum_values


l1 = LayerDense(3, 4)
l1.forward(inputs)
l2 = LayerDense(4, 4)
l2.forward(l1.outputs)
l3 = LayerDense(4, 2)
l3.forward(l2.outputs)

print(l3.outputs)
