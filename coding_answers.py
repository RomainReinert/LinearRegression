import numpy as np
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

# run the program the see the 3 different figures

# creating the matrix PHI of size N x (M+1) for a polynomial basis of a given order M
def phi_polynomial(X, order):
    X.flatten()
    N = X.shape[0]
    phi = np.zeros((N, order + 1))
    for i in range(N):
        for j in range(order + 1):
            phi[i][j] = X[i]**j
    return phi

# creating the matrix PHI of size N x (2M + 1) for a trigonometric basis of a given order M
def phi_trigonometric(X, order):
    X.flatten()
    N = X.shape[0]
    phi = np.zeros((N, 2*order + 1))
    for i in range(N):
        phi[i][0] = 1
        for j in range(1, order + 1):
            phi[i][2*j-1] = np.sin(2*np.pi*j*X[i])
            phi[i][2*j] = np.cos(2*np.pi*j*X[i])
    return phi

# calculating w for maximum likelihood
def max_lik_estimate(phi, y):
    w_ml = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(phi),phi)),np.transpose(phi)),y)
    return w_ml

# plotting the curves for poynomial basis of order 0, 1, 2, 3 and 11
def plot_polynomial():
    Xtest = np.linspace(-0.3,1.3,10000).reshape(-1,1)
    for i in (0,1,2,3,11):
        Phi = phi_polynomial(X, i)
        w_ml = max_lik_estimate(Phi, Y) 
        Phi_test = phi_polynomial(Xtest, i) 
        y_pred = Phi_test @ w_ml # computing predictions on [-0.3,1.3]
        plt.plot(Xtest, y_pred, label="Polynomial of order "+str(i))
        plt.legend()

# plotting the curves for trigonometric basis of order 1 and 11
def plot_trigonometric():
    Xtest = np.linspace(-1,1.2,10000).reshape(-1,1)
    for i in (1,11):
        Phi = phi_trigonometric(X, i)
        w_ml = max_lik_estimate(Phi, Y)
        Phi_test = phi_trigonometric(Xtest, i)
        y_pred = Phi_test @ w_ml
        plt.plot(Xtest, y_pred, label="Trigonometric of order "+str(i))
        plt.legend()
        

def question_c(): #leave-one-out cross validation
    for i in range(11): # for orders in [0,10]
        var = 0
        MSE = 0 
        for j in range(X.shape[0]): #cross-validation
            Xtest = X[j]
            Ytest = Y[j]
            Xtrain = np.concatenate((np.array((X[:j])),np.array((X[j+1:]))))
            Ytrain = np.concatenate((np.array((Y[:j])),np.array((Y[j+1:]))))
            Phi_train = phi_trigonometric(Xtrain, i)
            w_ml = max_lik_estimate(Phi_train, Ytrain)
            y_pred_train = Phi_train @ w_ml
            
            Phi_test = phi_trigonometric(Xtest, i)
            y_pred_test = Phi_test @ w_ml
            for l in range(Ytrain.shape[0]):
                var += (1/Ytrain.shape[0]) * (Ytrain[l] - y_pred_train[l])**2
            MSE += (1/1) * (Ytest[0] - y_pred_test[0])**2
            
        a=plt.scatter(i, var/X.shape[0], color='black', marker = 'x') # plotting the averaged results
        b=plt.scatter(i, MSE/X.shape[0], color='blue') 
    plt.legend((a,b), ("Variance estimator","Leave-one-out MSE"))
        
#plotting the 3 figures
plt.figure(1)
plt.ylim((-1.5,1.5))
plt.plot(X, Y, '+', label="data")
plot_polynomial()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

plt.figure(2)
plt.plot(X, Y, '+', label="data")
plot_trigonometric()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

plt.figure(3)
question_c()
plt.xlabel("$Basis' degrees$")
plt.ylabel("$Error$")

# # comparing the weights for polynomials of orders 4 and 11
# Phi = phi_polynomial(X, 4)
# w_ml = max_lik_estimate(Phi, Y)
# print("w_ml for a polynomial basis of order 4:")
# print(w_ml)
# Phi2 = phi_polynomial(X, 11)
# w_ml2 = max_lik_estimate(Phi2, Y)
# print("w_ml for a polynomial basis of order 11:")
# print(w_ml2)