# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

digits = load_digits()

print("Image Data: ", digits.data.shape)
print("Label Data: ", digits.target.shape)

plt.figure(figsize=(20, 4))
# zip takes iterables and aggregates them into a tuple and return it
# enumerate adds counter to an iterable and returns it
# so I'll have a list of (imageData, targetValue), with index
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)

# split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

logRegr = LogisticRegression()
# default solver is incredibly slow on MNIST dataset it's wise to change it
# default solver = liblinear => on MNIST 2893.1 seconds to run with 91.45% accuracy
# with solver = lbfgs => 52.86 seconds to run with 91.3% acc
# minor effect on accuracy, but a lot faster
# logRegr = LogisticRegression(solver = 'lbfgs')
logRegr.fit(x_train, y_train)


# making some predictions based on the model

print(y_test[0:10]) # print the first value from the test cases
print(logRegr.predict(x_test[0].reshape(1,-1))) # trying to predict the first value from the test cases
logRegr.predict(x_test[0:10]) # make predictions on the first 10 elements
predictions = logRegr.predict(x_test) # make predictions on the entire dataset

# determining the accuracy of the model
score = logRegr.score(x_test, y_test)
score # we have 0.9516908212560387 accuracy

# drawing the confusion matrix
cm = metrics.confusion_matrix(y_test, predictions) # generate confusion matrix
plt.figure(figsize=(9,9))
sb.heatmap(data=cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)

#  showing some misclassified images
index = 0
misclassifiedIndexes = []
for actual, predict in zip(y_test, predictions):
    if actual != predict:
        misclassifiedIndexes.append(index)
    index +=1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(x_test[badIndex], (8,8)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], y_test[badIndex]), fontsize = 15)
