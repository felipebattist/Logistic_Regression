import numpy as np
from sklearn.metrics import accuracy_score
import scipy

class Neuron():
    def __init__(self):
        self.weights = None
        self.b = None
        self.hist = None

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def initialize_with_zeros(self, dim):

        w = np.zeros(shape=(dim,1))
        b = 0

        return w,b

    def propagation(self, w, b, X, Y):

        #forward
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T, X) + b) #compute activation
        cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
        #backward
        dw = (1/m) * np.dot(X, (A - Y).T)
        db = (1/m) * np.sum(A - Y)

        grads = {'dw':dw, 'db':db}

        return grads, cost

    def update_parameters(self, w, b, X, Y, num_iterations = 100, learning_rate = 0.01, print_cost = False):

        costs = []

        grads, cost = self.propagation(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        # optimizing wheigths and the bias
        w = w - learning_rate * dw
        b = b - learning_rate * db

        for i in range(num_iterations-1):

            #forward propagation and backprop geting the dw and db
            grads, cost = self.propagation(w, b, X, Y)

            dw = grads['dw']
            db = grads['db']

            # optimizing wheigths and the bias
            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w, "b": b}

        grads = {"dw": dw, "db": db}

        return params, grads, costs

    def predict(self, w, b, X):

        m = X.shape[1]
        Y_pred = np.zeros((1,m))
        w = w = w.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            Y_pred[0, i] = 1 if A[0, i] > 0.5 else 0

        return Y_pred

    def fit(self, X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):

        w, b = self.initialize_with_zeros(X_train.shape[0])

        parameters, grads, costs = self.update_parameters(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        self.weights = parameters["w"]
        self.b = parameters["b"]

        hist = {'costs': costs, 'grads': grads, 'w':w, 'b':b, 'learning_rate':learning_rate, 'num_iterations': num_iterations}

        self.hist = hist

        return hist

    def evaluate(self, X_test, Y_test):
        try:
            Y_pred = self.predict(self.weights, self.b, X_test)
            acc = accuracy_score(Y_test[0], Y_pred[0])
            return acc
        except:
            raise ValueError('Before trying to evaluate the model you should fit the model with training data.')

