import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling


# Read data matrix X and labels y from text file.
def read_data(file_name):
    X = []
    y = []

    # Open the file in read mode
    file = open(file_name, "r")

    for line in file:
      values = line.split()

      # Convert the first value to float and append it to X (input features)
      X.append(float(values[0]))

      # Convert the second value to float and append it to y (target output)
      y.append(float(values[1]))

    # Convert X list to a NumPy array and reshape it to be a column vector        
    X = np.array(X).reshape(-1, 1)

    # Add a column of ones to X for the bias term
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Return the processed feature matrix X and target vector y as NumPy arrays
    # Because I didn't want to deal with them one by one later, 
    # I sent them directly as a numpy array.
    return X, np.array(y)


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lamda, epochs):

    w = np.zeros(X.shape[1])

    # Initialize an empty list to store the cost and train/test RMSE values at each epoch
    costs = []
    rmse_Train = []
    rmse_Values = []
    
    for i in range(epochs):
      
      # Calculate RMSE on training data
      rmse_Train.append(compute_rmse(Xtrain, ttrain, w))

      # Calculate RMSE on test data
      rmse_Values.append(compute_rmse(Xtest, ttest, w))
        
      # Compute the gradient of the cost function with respect to w
      grad = compute_gradient(X, y, w)

      # Update weights w by moving in the opposite direction of the gradient
      w -= lamda * grad

      # Calculate the cost with the updated weights
      cost = compute_cost(X, y, w)

      # Append the calculated cost to the costs list for tracking
      costs.append(cost)
    
    # Return the final weights and the list of costs over all epochs
    return w, costs, rmse_Train, rmse_Values


# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w):

    squared_Errors = 0
    
    for i in range(len(X)):
        
      # Calculate the prediction by taking the dot product of X[i] and w
      predictedData = X[i].dot(w)

      # Calculate the error as the difference between the prediction and the actual value y[i]
      errorDifference = predictedData - y[i]

      # Add the square of the error to the total squared error
      squared_Errors += errorDifference ** 2
    
    # Compute the mean squared error (MSE) by dividing the total squared error by the number of examples
    mse = squared_Errors / len(X)

    # Compute the root mean squared error (RMSE) by taking the square root of MSE
    rmse = np.sqrt(mse)

    # Return the computed RMSE
    return rmse


# Compute objective (cost) function.
def compute_cost(X, y, w):

    totalCost = 0
    
    for i in range(len(X)):
      
      # Calculate the prediction by taking the dot product of X[i] and w
      predictedData = X[i].dot(w)

      # Calculate the error as the difference between the prediction and the actual value y[i]
      errorDifference = predictedData - y[i]

      # Add the square of the error to the total cost
      totalCost += errorDifference ** 2
    
    # Compute the final cost by averaging and scaling the total cost
    cost = (1 / (2 * len(X))) * totalCost

    # Return the computed cost
    return cost


# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):

    grad = np.zeros(w.shape)
    
    for i in range(len(X)):

      # Calculate the prediction by taking the dot product of X[i] and w
      predictedData = X[i].dot(w)

      # Calculate the error as the difference between the prediction and the actual value y[i]
      errorDifference = predictedData - y[i]

      # Update the gradient by adding the error multiplied by the feature vector X[i]
      grad += errorDifference * X[i]

    # Divide the accumulated gradient by the number of examples to compute the average gradient
    grad /= len(X)

    return grad


## ======================= Main Program ======================= ##

# Read the training and test data.
Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")


# Calculate mean and standard deviation of training data (excluding bias term)
mean, std = scaling.mean_std(Xtrain[:, 1:])

# Standardize the training data features using the calculated mean and std
Xtrain[:, 1:] = scaling.standardize(Xtrain[:, 1:], mean, std)

# Standardize the test data features using the same mean and std from training data
Xtest[:, 1:] = scaling.standardize(Xtest[:, 1:], mean, std)

# Set the learning rate (lamda) and number of epochs for training

lamda = 0.1
epochs = 500


# Train the model using gradient descent and capture the final weights and cost values
w, costs, rmse_Train, rmse_Values = train(Xtrain, ttrain, lamda, epochs)



print("[My solution] ---> w =", w)

# I used this custom function to compare with my solution
w_functionSolution = np.linalg.pinv(Xtrain.T.dot(Xtrain)).dot(Xtrain.T).dot(ttrain)
print("[linalg.pinv solution] ---> w =", w_functionSolution)



# Compute RMSE for the training data with the trained weights
rmse_train = compute_rmse(Xtrain, ttrain, w)

# Compute RMSE for the test data with the trained weights
rmse_test = compute_rmse(Xtest, ttest, w)

print("[RMSE] Training Data:", rmse_train)
print("[RMSE] Test Data:", rmse_test)




# Plot cost function values over epochs to visualize convergence
plt.plot(range(epochs), costs, color="purple")
plt.xlabel("Epochs")          # Label x-axis
plt.ylabel("J(w)")            # Label y-axis
plt.title("Epoch -vs- J(w)")  # Title for the plot
plt.show()                    # Display the cost function plot


# Plot RMSE values over epochs
plt.plot(range(epochs), rmse_Train, color="blue", label="Training RMSE")
plt.plot(range(epochs), rmse_Values, color="green", label="Test RMSE")
plt.xlabel("Epochs")          # Label x-axis
plt.ylabel("RMSE")            # Label y-axis
plt.title("RMSE Values")      # Title for the plot
plt.legend()
plt.show()                    # Display the RMSE plot



# Plot the training and test data along with the regression line
plt.scatter(Xtrain[:, 1], ttrain, color="blue", label="Training Data")          # blue circles training data
plt.scatter(Xtest[:, 1], ttest, color="green", marker="x", label="Test Data")   # green x test datas
plt.plot(Xtrain[:, 1], Xtrain.dot(w), color="red", label="Regression Line")     # red regression line -> solution
plt.xlabel("Standardized Values")
plt.ylabel("Price of the Houses")
plt.legend()
plt.show




#######  Written by  #######
######    Zer0-Bug    ######