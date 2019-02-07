import numpy as np
import pandas as pd

class neuralnetwork:
    # initialize parameters
    def __init__(self,num_sample,num_hidden_layer_units,num_input_feature,num_output_feature,bias,learningrate):
        self.learningrate=learningrate # learning rate
        self.num_sample = num_sample  # number of the samples
        self.num_hidden_layer_units = num_hidden_layer_units # a list indicating the number of hidden layers and how many neurons for each layer
        self.num_hidden_layer = len(num_hidden_layer_units) # the number of the hidden layers
        self.weight_layer = []  # initialize the weight matrix
        self.bias = bias   # bias for the input layer and for the hidden layers
        assert len(self.bias) == self.num_hidden_layer+1, "The length of the biases should equal the length of the hidden layers plus 1!!!"
        self.units = [num_input_feature] + num_hidden_layer_units + [num_output_feature]
        # randomly initialize the weight matrix
        for num_layer in range(0,len(self.units)-1):
            temp_bias = np.zeros((1,self.units[num_layer+1]))
            temp_bias.fill(self.bias[num_layer])
            self.weight_layer.append(np.concatenate((np.random.randn(self.units[num_layer+1],
                                                self.units[num_layer]),temp_bias.T),axis = 1))

    # the activation function: sigmoid
    # could be replaced by whatever you want
    def sigmoid(self,x):
        y = 1/(1+np.exp(-x))
        return y

    # train function requires the training data, accuracy and iteration limit
    def train(self, input, output,accuracy, iteration_limit):
        go_on = True
        i = 0 # iteration indicator
        error = []  # error list recoding errors for all iterations
        iteration = []
        while go_on:
            iteration.append(i)
            error_acc = 0   # initialize error for every iteration
            stderror = []   # standard error
            sample_Z = []   # Z value of neurons for every sample
            sample_a = []   # sigmoid value of Z for every sample
            # loop all samples
            for sample_iter in range(self.num_sample):
                Zlayer = []
                a = []
                a.append(input[sample_iter])    # place the input as the first a value
                # forward computing Z and a values for all layers
                for forward_layer in range(len(self.units)-1):
                    Zlayer.append(np.matmul(self.weight_layer[forward_layer], np.concatenate((a[forward_layer],[1]))))
                    a.append(self.sigmoid(Zlayer[forward_layer]))
                # standard error by computing the distance between output layer and true output
                stderror.append(a[len(a)-1]-output[sample_iter])
                # loss function defined as the sum of the least square
                # can be replaced by other functions like cross-entropy
                error_acc += (1/2*np.sum((stderror[sample_iter])**2))
                sample_a.append(a)  # store a values for updating
                sample_Z.append(Zlayer) # store Z values for updating
            error.append(error_acc) # store error for this iteration

            # backward propagate errors to update the weight matrix
            i += 1
            if error[i-1] <accuracy or i >iteration_limit:
                go_on = False
            else:
                for sample_iter in range(self.num_sample):
                    delta = stderror[sample_iter]  # delta: standard error defined as the derivative of the loss function
                    # initialize g'(z): the derivative of the activation function at the output layer
                    deriv_sigmoid =  sample_a[sample_iter][len(sample_a[sample_iter])-1] *\
                                 (1 - sample_a[sample_iter][len(sample_a[sample_iter])-1])
                    # backward propagation
                    for backward_layer in list(reversed(range(len(self.units)-1))):
                        temp_weight = self.weight_layer[backward_layer] # store the temporary kth matrix
                        # update the kth matrix
                        self.weight_layer[backward_layer][:,:-1] += - self.learningrate/self.num_sample *\
                            np.outer(delta * deriv_sigmoid, sample_a[sample_iter][backward_layer])
                        # update delta and derivative of the activation function
                        delta = np.dot(temp_weight[:,:-1].T,delta * deriv_sigmoid)
                        deriv_sigmoid = sample_a[sample_iter][backward_layer] *\
                                 (1 - sample_a[sample_iter][backward_layer])
        return sample_a
        self.learningspeed = {'error':error,'iteration':iteration}
        self.lsdf = pd.DataFrame(self.learningspeed)


inputx = np.array(([0.2, 0.5, 0.5,0.3], [0.1, 0.2, 0.15,0.1],[0.7,0.9,0.4,0.8]))
outputy = np.array(([0.5, 0.4], [0.9, 0.1],[0.3,0.5]))

nn=neuralnetwork(num_sample=inputx.shape[0],num_hidden_layer_units=[15,14,15,16],num_input_feature=inputx.shape[1],
             num_output_feature=outputy.shape[1], bias=[0.1,0.2,0.3,0.4,0.4],learningrate=0.2)


acc = 1e-7
iter = 20000
out_a = nn.train(inputx,outputy,accuracy = acc,iteration_limit=iter)

# nn.lsdf.plot(x='iteration',y='error')