"""
AASHWIN VATS
"""

from __future__ import division
from __future__ import print_function
from scipy import special as sp
import math

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
import matplotlib.pyplot as plt

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
    # DEFINE __init function
        self.W = W
        self.b = b
	
    def forward(self, x):
	# DEFINE forward function
		#linear func.: weighted sum of inputs plus bias
        self.X = x
        self.fx = np.dot(self.X, self.W) + self.b
        return self.fx

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):  
	# DEFINE backward function
        dW = np.dot(self.X.T, grad_output) + l2_penalty*self.W 
        db = np.sum(grad_output, axis =0)
        dA = np.dot(grad_output, self.W.T)
        return dA, dW, db


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        self.AF1 = (x*(x>0))
        return self.AF1

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ): 
    # DEFINE backward function
        dZ1 = np.zeros((grad_output.shape))

        dZ1[self.AF1 < 0] = 0
        dZ1[self.AF1 == 0] = np.random.uniform(0.01, 1)
        dZ1[self.AF1 > 0] = 1
        
        dZ1 = np.multiply(dZ1, grad_output)
        return dZ1


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form

class SigmoidCrossEntropy(object):
	
    def forward(self, x):
        self.AF2 = sp.expit(x)  
        return self.AF2
		
    def backward(self, y_batch, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        dZ2 = self.AF2 - y_batch
        return dZ2


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network

        self.input_dims = input_dims
        self.hidden_units = hidden_units

        self.W1 = np.random.randn(input_dims, hidden_units) * 0.1  
        self.W2 = np.random.randn(hidden_units, 1) * 0.1
        self.b1 = np.zeros((1, hidden_units))* 0.1  
        self.b2 = np.zeros((1, 1))* 0.1  

        self.NdW1 = np.zeros((self.W1.shape))
        self.NdW2 = np.zeros((self.W2.shape))
        self.Ndb1 = np.zeros((self.b1.shape))
        self.Ndb2 = np.zeros((self.b2.shape))

    def forwardfeed(self, x_batch):
        self.linearF1 = LinearTransform(self.W1, self.b1)
        self.Z1 = self.linearF1.forward(x_batch)
        self.relu = ReLU()
        self.A1= self.relu.forward(self.Z1)

        self.linearF2 = LinearTransform(self.W2, self.b2)
        self.Z2 = self.linearF2.forward(self.A1)
        self.sigmoid = SigmoidCrossEntropy()
        self.A2 = self.sigmoid.forward(self.Z2)

    def backwardfeed(self, y_batch, l2_penalty):

        dZ2 = self.sigmoid.backward(y_batch)
        dA1, self.dW2, self.db2 = self.linearF2.backward(dZ2, l2_penalty)
        m = y_batch.shape[0]
        self.dW2, self.db2 =  (1.0/m)*self.dW2, (1.0/m)*self.db2

        dZ1 = self.relu.backward(dA1)
        dA0, self.dW1, self.db1 = self.linearF1.backward(dZ1, l2_penalty)
        self.dW1, self.db1 = (1.0/m) * self.dW1, (1.0/m) * self.db1

    def calcLoss(self, y_batch):
        self.loss =  (np.multiply(y_batch, np.log(self.A2)) + np.multiply((1 - y_batch), np.log(1-self.A2))) 
        self.loss = -1*np.mean(self.loss, axis = 0) 
        return self.loss

    def updateParam(self, learning_rate, momentum):

        # updating weights
        self.NdW1 = momentum*self.NdW1 - learning_rate*self.dW1
        self.NdW2 = momentum*self.NdW2 - learning_rate*self.dW2
        self.W1 = self.W1 + self.NdW1
        self.W2 = self.W2 + self.NdW2
        
        # updating bias
        self.Ndb1 = momentum*self.Ndb1 - learning_rate*self.db1
        self.Ndb2 = momentum*self.Ndb2 - learning_rate*self.db2
        self.b1 = self.b1 + self.Ndb1
        self.b2 = self.b2 + self.Ndb2

    def predict(self):
        predict = np.zeros((self.A2.shape))
        predict[self.A2 <= 0.5] = 0 
        predict[self.A2 > 0.5] = 1
        return predict

    def calcError(self, y_batch):
        result = (self.predict() == y_batch)
        self.errorCount = len((np.where(result==False))[0])

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty
    ):
	# INSERT CODE for training the network

        self.forwardfeed(x_batch)
        loss = self.calcLoss(y_batch)
        self.backwardfeed(y_batch, l2_penalty)
        self.updateParam(learning_rate, momentum)
        self.calcError(y_batch)

        return self.W1, self.W2, self.b1, self.b2, self.A2, np.squeeze(self.loss), self.errorCount

    def evaluate(self, x, y):
	# INSERT CODE for testing the network
        self.forwardfeed(x)
        loss = self.calcLoss(y)
        self.calcError(y)
        return self.loss, self.errorCount

# creating mini batches for stochiastic gradient descent:
def createMiniBatches(X, Y, num_batches = None, batchSize= 16 ):
    num_examples = X.shape[0]
    mini_batches = []
    permutations = list(np.random.permutation(num_examples))
    x_shuffled = X[permutations, :]
    y_shuffled = Y[permutations, :]

    if num_batches == None:
        num_batches = math.floor(num_examples/batchSize)

        for i in range(num_batches):
            x_batch = x_shuffled[batchSize*i:batchSize*(i+1),:]
            y_batch = y_shuffled[batchSize*i:batchSize*(i+1),:]
            mini_batches.append((x_batch, y_batch))

        if num_examples%batchSize != 0:
            x_batch = x_shuffled[batchSize*num_batches:,:]
            y_batch = y_shuffled[batchSize*num_batches:,:]
            mini_batches.append((x_batch, y_batch))
            num_batches += 1

    else:
        batchSize = math.ceil(num_examples/num_batches)   

        for i in range(num_batches-1):
            x_batch = x_shuffled[batchSize*i:batchSize*(i+1),:]
            y_batch = y_shuffled[batchSize*i:batchSize*(i+1),:]
            mini_batches.append((x_batch, y_batch))

        x_batch = x_shuffled[batchSize*(num_batches-1):,:]
        y_batch = y_shuffled[batchSize*(num_batches-1):,:]
        mini_batches.append((x_batch, y_batch))   

    return mini_batches, num_batches

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data'] #convert to np array
    train_x = np.array(train_x, dtype = 'float64') 
    train_y = data[b'train_labels']
    test_x = data[b'test_data']
    test_y = data[b'test_labels']

    # normalizing datasets 
    train_x = train_x / 255.0
    test_x = test_x / 255.0
	
    num_examples, input_dims = train_x.shape
    test_examples = test_x.shape[0]
	

    # INSERT YOUR CODE HERE

	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES 
    
    num_epochs = 100
    num_batches = 1000
    hidden_units = 600
    learning_rate=0.001
    momentum = 0.8
    l2_penalty = 0.001
    cnt_error_train = []
    cnt_loss_train = []
    cnt_error_test = []
    cnt_loss_test = []  

    mlp = MLP(input_dims, hidden_units)

    for epoch in range(num_epochs):
    # INSERT YOUR CODE FOR EACH EPOCH HERE
        
        total_error = 0
        total_loss = 0.0

        learning_rate = learning_rate*math.pow(0.5, math.floor((1+epoch)/10.0))
        mini_batches, num_batches = createMiniBatches(train_x, train_y, None)

        for b in range(num_batches):

            batch_x = mini_batches[b][0]
            batch_y = mini_batches[b][1]
            
            W1, W2, b1, b2, A2, loss, errorCount = mlp.train(batch_x, batch_y, learning_rate, momentum, l2_penalty)

            total_error += errorCount
            total_loss += loss

            # MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    loss,
                ),
                end='',
            )
            sys.stdout.flush()

        total_loss_test, total_error_test = mlp.evaluate(test_x, test_y)

        cnt_error_train.append(total_error)
        cnt_loss_train.append(total_loss/num_batches)
        cnt_error_test.append(total_error_test)
        cnt_loss_test.append(total_loss_test/num_batches)

    #  train data
    mlp.forwardfeed(train_x)
    train_loss = mlp.calcLoss(train_y)
    predict = mlp.predict()
    result = (predict == train_y)
    errorCount = len((np.where(result==False))[0])
    train_accuracy = ((num_examples- errorCount)/num_examples)

    train_misclass_rate = (errorCount/num_examples)*100.
    print("\nTraining Misclassification Rate(%):", train_misclass_rate)



    # test data
    mlp.forwardfeed(test_x)
    test_loss = mlp.calcLoss(test_y)
    predict = mlp.predict()
    result = (predict == test_y)
    errorCount = len((np.where(result==False))[0])
    test_accuracy = ((test_examples-errorCount)/test_examples)

    test_misclass_rate = (errorCount/test_examples)*100.
    print("Testing Misclassification Rate(%):", test_misclass_rate)
    


    # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    print('\n    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
        train_loss[0],
        100.*train_accuracy,
    ))

    print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
        test_loss[0],
        100. *test_accuracy,
    ))
    







#--------------------------------Tuning Parameters------------------------------#
# ------------------------------Code to generate graphs-------------------------#
# Comment the above code and uncomment the following to generate the figures
# code not optimized, but repeated

    # UNCOMMENT:To get accuracy vs batch size graph

    # batchSize = [8, 16, 32, 64, 128]
    # test_accuracy_graph = []

    # for a in range(len(batchSize)):
    #     # print("minibatch", batchSize[a])
    #     num_epochs = 10
    #     num_batches = 1000
    #     hidden_units = 50
    #     learning_rate=0.001
    #     momentum = 0.8
    #     l2_penalty = 0.001
    #     cnt_error_train = []
    #     cnt_loss_train = []
    #     cnt_error_test = []
    #     cnt_loss_test = []  

    #     mlp = MLP(input_dims, hidden_units)

    #     for epoch in range(num_epochs):
    # 	# INSERT YOUR CODE FOR EACH EPOCH HERE
            
    #         total_error = 0
    #         total_loss = 0.0

    #         learning_rate = learning_rate*math.pow(0.5, math.floor((1+epoch)/10.0))
    #         mini_batches, num_batches = createMiniBatches(train_x, train_y, None, batchSize[a])

    #         for b in range(num_batches):

    #             batch_x = mini_batches[b][0]
    #             batch_y = mini_batches[b][1]
                
    #             W1, W2, b1, b2, A2, loss, errorCount = mlp.train(batch_x, batch_y, learning_rate, momentum, l2_penalty)

    #             total_error += errorCount
    #             total_loss += loss

    # 			# MAKE SURE TO UPDATE total_loss
    #             print(
    #                 '\r[Epoch {}, mb {}]    Avg.loss = {:.3f}'.format(
    #                     epoch + 1,
    #                     b + 1,
    #                     loss,
    #                 ),
    #                 end='',
    #             )
    #             sys.stdout.flush()

    #         total_loss_test, total_error_test = mlp.evaluate(test_x, test_y)

    #         cnt_error_train.append(total_error)
    #         cnt_loss_train.append(total_loss/num_batches)
    #         cnt_error_test.append(total_error_test)
    #         cnt_loss_test.append(total_loss_test/num_batches)

    #     #  train data
    #     mlp.forwardfeed(train_x)
    #     train_loss = mlp.calcLoss(train_y)
    #     predict = mlp.predict()
    #     result = (predict == train_y)
    #     errorCount = len((np.where(result==False))[0])
    #     train_accuracy = ((num_examples- errorCount)/num_examples)

        

    #     # test data
    #     mlp.forwardfeed(test_x)
    #     test_loss = mlp.calcLoss(test_y)
    #     predict = mlp.predict()
    #     result = (predict == test_y)
    #     errorCount = len((np.where(result==False))[0])
    #     test_accuracy = ((test_examples-errorCount)/test_examples)


    # 	# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    #     print('\n    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
    #         train_loss[0],
    #         100.*train_accuracy,
    #     ))

    #     print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
    #         test_loss[0],
    #         100. *test_accuracy,
    #     ))

    #     train_accuracy = [100 * (num_examples - errorCount)/ num_examples for errorCount in cnt_error_train]
    #     test_accuracy = [100 * (test_examples - errorCount)/ test_examples for errorCount in cnt_error_test]
    #     test_accuracy_graph.append(test_accuracy[1])

    # # print(test_accuracy_graph)
    
    # plt.figure()

    # plt.xlabel('BatchSize')
    # plt.ylabel('Test Accuracy')
    # plt.title('Accuracy of testing data vs Batch Size')
    # plt.plot( batchSize, test_accuracy_graph)
    # plt.show()

    # UNCOMMENT: To plot Accuracy with different learning rate, keeping batch size = 32

    # learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    # test_accuracy_graph = []

    # for a in range(len(learning_rate)):
    #     print("learning rate", learning_rate[a])
    #     num_epochs = 10
    #     num_batches = 1000
    #     hidden_units = 50
    #     momentum = 0.8
    #     l2_penalty = 0.001
    #     cnt_error_train = []
    #     cnt_loss_train = []
    #     cnt_error_test = []
    #     cnt_loss_test = []  

    #     mlp = MLP(input_dims, hidden_units)

    #     for epoch in range(num_epochs):
    #     # INSERT YOUR CODE FOR EACH EPOCH HERE
            
    #         total_error = 0
    #         total_loss = 0.0

    #         learning_rate[a] = learning_rate[a] *math.pow(0.5, math.floor((1+epoch)/10.0))
    #         mini_batches, num_batches = createMiniBatches(train_x, train_y, None)

    #         for b in range(num_batches):

    #             batch_x = mini_batches[b][0]
    #             batch_y = mini_batches[b][1]
                
    #             W1, W2, b1, b2, A2, loss, errorCount = mlp.train(batch_x, batch_y, learning_rate[a] , momentum, l2_penalty)

    #             total_error += errorCount
    #             total_loss += loss

    #             # MAKE SURE TO UPDATE total_loss
    #             print(
    #                 '\r[Epoch {}, mb {}]    Avg.loss = {:.3f}'.format(
    #                     epoch + 1,
    #                     b + 1,
    #                     loss,
    #                 ),
    #                 end='',
    #             )
    #             sys.stdout.flush()

    #         total_loss_test, total_error_test = mlp.evaluate(test_x, test_y)

    #         cnt_error_train.append(total_error)
    #         cnt_loss_train.append(total_loss/num_batches)
    #         cnt_error_test.append(total_error_test)
    #         cnt_loss_test.append(total_loss_test/num_batches)

    #     #  train data
    #     mlp.forwardfeed(train_x)
    #     train_loss = mlp.calcLoss(train_y)
    #     predict = mlp.predict()
    #     result = (predict == train_y)
    #     errorCount = len((np.where(result==False))[0])
    #     train_accuracy = ((num_examples- errorCount)/num_examples)

    #     # test data
    #     mlp.forwardfeed(test_x)
    #     test_loss = mlp.calcLoss(test_y)
    #     predict = mlp.predict()
    #     result = (predict == test_y)
    #     errorCount = len((np.where(result==False))[0])
    #     test_accuracy = ((test_examples-errorCount)/test_examples)


    #     # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    #     print('\n    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
    #         train_loss[0],
    #         100.*train_accuracy,
    #     ))

    #     print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
    #         test_loss[0],
    #         100. *test_accuracy,
    #     ))

    #     train_accuracy = [100 * (num_examples - errorCount)/ num_examples for errorCount in cnt_error_train]
    #     test_accuracy = [100 * (test_examples - errorCount)/ test_examples for errorCount in cnt_error_test]
    #     test_accuracy_graph.append(test_accuracy[1])

    # # print(test_accuracy_graph)
    
    # plt.figure()

    # plt.xlabel('Learning Rate')
    # plt.ylabel('Test Accuracy')
    # plt.title('Accuracy of testing data vs Learning Rate')
    # plt.plot( learning_rate, test_accuracy_graph)
    # plt.show()
    # plt.savefig('AccuracyVsLearningRate.png')


    # UNCOMMMENT: To plot graph for test accuracy with different number of hidden units
    # hidden_units = [10, 50, 100, 150, 200, 500, 1000, 1500, 2000, 2500 ,3000]
    # test_accuracy_graph = []

    # for a in range(len(hidden_units)):
    #     print("hidden_units", hidden_units[a])
    #     num_epochs = 10
    #     num_batches = 1000
    #     # hidden_units = 50
    #     learning_rate=0.001
    #     momentum = 0.8
    #     l2_penalty = 0.001
    #     cnt_error_train = []
    #     cnt_loss_train = []
    #     cnt_error_test = []
    #     cnt_loss_test = []  

    #     mlp = MLP(input_dims, hidden_units[a])

    #     for epoch in range(num_epochs):
    #   # INSERT YOUR CODE FOR EACH EPOCH HERE
            
    #         total_error = 0
    #         total_loss = 0.0

    #         learning_rate = learning_rate*math.pow(0.5, math.floor((1+epoch)/10.0))
    #         mini_batches, num_batches = createMiniBatches(train_x, train_y, None)

    #         for b in range(num_batches):

    #             batch_x = mini_batches[b][0]
    #             batch_y = mini_batches[b][1]
                
    #             W1, W2, b1, b2, A2, loss, errorCount = mlp.train(batch_x, batch_y, learning_rate, momentum, l2_penalty)

    #             total_error += errorCount
    #             total_loss += loss

    #           # MAKE SURE TO UPDATE total_loss
    #             print(
    #                 '\r[Epoch {}, mb {}]    Avg.loss = {:.3f}'.format(
    #                     epoch + 1,
    #                     b + 1,
    #                     loss,
    #                 ),
    #                 end='',
    #             )
    #             sys.stdout.flush()

    #         total_loss_test, total_error_test = mlp.evaluate(test_x, test_y)

    #         cnt_error_train.append(total_error)
    #         cnt_loss_train.append(total_loss/num_batches)
    #         cnt_error_test.append(total_error_test)
    #         cnt_loss_test.append(total_loss_test/num_batches)

    #     #  train data
    #     mlp.forwardfeed(train_x)
    #     train_loss = mlp.calcLoss(train_y)
    #     predict = mlp.predict()
    #     result = (predict == train_y)
    #     errorCount = len((np.where(result==False))[0])
    #     train_accuracy = ((num_examples- errorCount)/num_examples)

        

    #     # test data
    #     mlp.forwardfeed(test_x)
    #     test_loss = mlp.calcLoss(test_y)
    #     predict = mlp.predict()
    #     result = (predict == test_y)
    #     errorCount = len((np.where(result==False))[0])
    #     test_accuracy = ((test_examples-errorCount)/test_examples)


    #   # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    #     print('\n    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
    #         train_loss[0],
    #         100.*train_accuracy,
    #     ))

    #     print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
    #         test_loss[0],
    #         100. *test_accuracy,
    #     ))

    #     train_accuracy = [100 * (num_examples - errorCount)/ num_examples for errorCount in cnt_error_train]
    #     test_accuracy = [100 * (test_examples - errorCount)/ test_examples for errorCount in cnt_error_test]
    #     test_accuracy_graph.append(test_accuracy[1])

    # plt.figure()

    # plt.xlabel('Hidden Units')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy of testing data vs Hidden Units')
    # plt.plot( hidden_units, test_accuracy_graph)
    # plt.show()

