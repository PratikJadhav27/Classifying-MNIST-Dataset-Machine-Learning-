# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.datasets import mnist
import cv2

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the threshold value
threshold_value = 127

# Threshold the images
x_train_thresholded = np.where(x_train > threshold_value, 1, 0)
x_test_thresholded = np.where(x_test > threshold_value, 1, 0)

### BOUNDING BOX

# Define a function to construct a bounding box around a given image
def construct_bounding_box(image):
    # Compute row-wise and column-wise sums of the thresholded image
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

    # Find the range of ink pixels along each row and column
    row_nonzero = np.nonzero(row_sums)[0]
    col_nonzero = np.nonzero(col_sums)[0]
    
    # If there are no ink pixels, return a black image of size 20x20
    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        return np.zeros((20, 20))
    
    # Otherwise, compute the row and column ranges of the ink pixels
    row_range = row_nonzero[[0, -1]]
    col_range = col_nonzero[[0, -1]]

    # Compute the center of the ink pixel ranges
    row_center = (row_range[0] + row_range[-1]) / 2
    col_center = (col_range[0] + col_range[-1]) / 2

    # Compute starting and ending indices for the bounding box
    row_start = int(np.clip(row_center - 9, 0, image.shape[0] - 20))
    row_end = row_start + 20
    col_start = int(np.clip(col_center - 9, 0, image.shape[1] - 20))
    col_end = col_start + 20

    # Extract the bounding box from the image
    bounding_box = image[row_start:row_end, col_start:col_end]

    return bounding_box

# Define a function to construct a stretched bounding box around a given image
def construct_bounding_box_stretched(image):
    # Compute row-wise and column-wise sums of the thresholded image
    row_sums = np.sum(image, axis=1)
    col_sums = np.sum(image, axis=0)

    # Find the range of ink pixels along each row and column
    row_nonzero = np.nonzero(row_sums)[0]
    col_nonzero = np.nonzero(col_sums)[0]
    
    # If there are no ink pixels, return a black image of size 20x20
    if len(row_nonzero) == 0 or len(col_nonzero) == 0:
        return np.zeros((20, 20))

    # Otherwise, compute the row and column ranges of the ink pixels
    row_range = row_nonzero[[0, -1]]
    col_range = col_nonzero[[0, -1]]
    row_start, row_end = row_range[0], row_range[-1]
    col_start, col_end = col_range[0], col_range[-1]

    # Stretch the extracted image to 20x20 dimensions
    image = image[row_start:row_end, col_start:col_end]
    image = resize(image, (20, 20))

    return image

# Initialize numpy arrays to hold bounding box images
x_train_bounding_box = np.zeros((len(x_train_thresholded), 20, 20))
x_train_bounding_box_stretched = np.zeros((len(x_train_thresholded), 20, 20))

# Loop through all training images and apply function to construct bounding box and stretched bounding box
for i in range(len(x_train_thresholded)):
    x_train_bounding_box[i] = construct_bounding_box(x_train_thresholded[i])
    x_train_bounding_box_stretched[i] = construct_bounding_box_stretched(x_train_thresholded[i])

# Initialize numpy arrays to hold bounding box images for test set
x_test_bounding_box = np.zeros((len(x_test_thresholded), 20, 20))
x_test_bounding_box_stretched = np.zeros((len(x_test_thresholded), 20, 20))

# Loop through all test images and apply function to construct bounding box and stretched bounding box
for i in range(len(x_test_thresholded)):
    x_test_bounding_box[i] = construct_bounding_box(x_test_thresholded[i])
    x_test_bounding_box_stretched[i] = construct_bounding_box_stretched(x_test_thresholded[i])

# Plot example images of original, thresholded, bounding box, and stretched bounding box images for the first 5 images in training set
fig, axs = plt.subplots(4, 5, figsize=(10, 5))
axs = axs.ravel()

i = [10, 12, 34, 5, 7]
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 8))
for j in range(5):
    axs[0, j].imshow(x_train[i[j]], cmap='gray')
    axs[0, j].set_title('Original Image')
    axs[1, j].imshow(x_train_thresholded[i[j]], cmap='gray')
    axs[1, j].set_title('Thresholded Image')
    axs[2, j].imshow(x_train_bounding_box[i[j]], cmap='gray')
    axs[2, j].set_title('Bounding Box Image')
    axs[3, j].imshow(x_train_bounding_box_stretched[i[j]], cmap='gray')
    axs[3, j].set_title('Stretch Bounding Box Image')
plt.show()

# Plot example images of original, thresholded, bounding box, and stretched bounding box images for 5 images of the digit 7 in training set
fig, axs = plt.subplots(4, 5, figsize=(10, 5))
axs = axs.ravel()
count = 0
for i in range(len(x_train)):
    if y_train[i] == 7 and count < 5:
        axs[count].imshow(x_train[i], cmap='gray')
        axs[count].set_title('Original Image')
        axs[count+5].imshow(x_train_thresholded[i], cmap='gray')
        axs[count+5].set_title('Thresholded Image')
        axs[count+10].imshow(x_train_bounding_box[i], cmap='gray')
        axs[count+10].set_title('Bounding Box Image')
        axs[count+15].imshow(x_train_bounding_box_stretched[i], cmap='gray')
        axs[count+15].set_title('Stretch Bound Box')
        count +=1

plt.show()





##############################################################################################
from sklearn.neighbors import KNeighborsClassifier

# KNN for thresholded image 

# Reshape the training and test images to 1D arrays
x_train_thresholded_1d = x_train_thresholded.reshape(len(x_train_thresholded), -1)
x_test_thresholded_1d = x_test_thresholded.reshape(len(x_test_thresholded), -1)

# Initialize a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(x_train_thresholded_1d, y_train)

# Compute the accuracy of the classifier on the test data
accuracy = knn.score(x_test_thresholded_1d, y_test)
print('Accuracy of KNN classifier on test set: {:.4f}'.format(accuracy))

# KNN for bounding box image

# Reshape the training and test images to 1D arrays
x_train_bounding_box_1d = x_train_bounding_box.reshape(len(x_train_bounding_box), -1)
x_test_bounding_box_1d = x_test_bounding_box.reshape(len(x_test_bounding_box), -1)

# Initialize a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(x_train_bounding_box_1d, y_train)

# Compute the accuracy of the classifier on the test data
accuracy = knn.score(x_test_bounding_box_1d, y_test)
print('Accuracy of KNN classifier on test set: {:.4f}'.format(accuracy))

# KNN for stretched bounding box image

# Reshape the training and test images to 1D arrays
x_train_bounding_box_stretched_1d = x_train_bounding_box_stretched.reshape(len(x_train_bounding_box_stretched), -1)
x_test_bounding_box_stretched_1d = x_test_bounding_box_stretched.reshape(len(x_test_bounding_box_stretched), -1)

# Initialize a KNN classifier

knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(x_train_bounding_box_stretched_1d, y_train)

# Compute the accuracy of the classifier on the test data
accuracy = knn.score(x_test_bounding_box_stretched_1d, y_test)

print('Accuracy of KNN classifier on test set: {:.4f}'.format(accuracy))

