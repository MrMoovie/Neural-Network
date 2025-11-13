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

class Loss:
    def calculate(self, output, y): # output(matrix) - from the model, y(matrix/vector) - intended values
        sample_losses = self.forward(output, y) # depends on the desired loss function
        return sample_losses

    def accuracy(self, output, y):
        predictions = np.argmax(output, axis=1)
        if len(y.shape) == 2:
            y=np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        global correct_confidences
        nSamples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1: #e.g. [1,1,0,2]
            # fishing out the predictions of the -supposed to be- correct values,
            # to calculate later the loss.
            # by choosing each the value, by indexes stored in 'y_true'.
            # e.g. y_true = [1, 0, 0] - we'll choose the 1st index from the 1st batch, 0th from 2nd etc.
            correct_confidences = y_pred_clipped[range(nSamples), y_true]

        elif len(y_true.shape) == 2: #e.g. [[0,1,0], [0,1,0], [1,0,0], [0,0,1]]
            # in that case, we'll multiply the matrixes, which will automatically give
            # a matrix with only the wanted values, and 0's.
            # then we'll sum the values in axis=1, and get a vector of the wanted values, representing each batch.
            correct_confidences = np.sum(y_pred_clipped*y_true, axis =1)

        negative_log_likelihood = -np.log(correct_confidences)
        return np.mean(negative_log_likelihood)


l1 = LayerDense(3, 4) # init layer 1 (input)
l1.forward(inputs)
l2 = LayerDense(4, 4) # init layer 2
l2.forward(l1.outputs)
l3 = LayerDense(4, 2) # init layer 3 (output)
l3.forward(l2.outputs)

print(l3.outputs)
