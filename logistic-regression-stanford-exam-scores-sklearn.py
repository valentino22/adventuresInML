import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


# The first two columns contains the exam scores and the third column contains the success of exam.
dataset = pd.read_csv('data/ex2data1.txt')

# get the correct columns from the dataset
x = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

# split the data into training and validation set
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=20, random_state=0)

# normalize the data for faster computation
normalizer = StandardScaler()
x_normalized = normalizer.fit_transform(x)

# run Logistic Regression and create the model
logRegClassifier = LogisticRegression(random_state=0)
logRegClassifier.fit(x_normalized, y)

# make a prediction for the validation set
prediction = logRegClassifier.predict(normalizer.transform(x))
print(prediction)

# create a confusion matrix to see how accurate it is
cm = confusion_matrix(y, prediction)
print('Confusion Matrix:\n', cm)

# print the score of the model based on the validation set
accuracy = logRegClassifier.score(x_normalized, y)
print('The accuracy of the model is:', '{0:.2f}'.format(accuracy))

# make a prediction for an example student exam scores (he got 45 for one and 85 for the other exam)
predict_exam = logRegClassifier.predict(normalizer.transform([[45, 85]]))
# the prediction is 1 so this student should pass the class
print('Prediction if student will pass the class:', predict_exam[0])

# print the probability of student passing based on example
predict_pass_probability = logRegClassifier.predict_proba(normalizer.transform([[45, 85]]))
print('Probability that the student will pass the class:', predict_pass_probability[0])
# [0.26082152 0.73917848] [student will not pass, the student passes]


# plot the results
X_set, y_set = normalizer.transform(x), y
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, logRegClassifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=[ListedColormap(('red', 'green'))(i)], label=j)

plt.title('Classifier (Test set)')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.legend()
plt.show()
