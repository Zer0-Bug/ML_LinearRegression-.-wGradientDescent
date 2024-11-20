import numpy as np

# Compute the sample mean and standard deviations for each feature (column)
# across the training examples (rows) from the data matrix X.

def mean_std(X):

  mean = np.mean(X, axis=0)   # 'axis=0' means calculating the mean across rows (for each feature)

  std = np.std(X, axis=0)     # 'axis=0' means calculating std across rows (for each feature)
  
  return mean, std



# Standardize the features of the examples in X by subtracting their mean and 
# dividing by their standard deviation, as provided in the parameters.
def standardize(X, mean, std):

  # Subtract the mean from each feature and divide by the standard deviation to normalize
  return (X - mean) / std





#######  Written by  #######
######    Zer0-Bug    ######