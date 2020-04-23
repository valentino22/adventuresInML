import matplotlib.pyplot as plt
from matplotlib import style # to be able to use different styles
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression

# use a specific style, there are lots of styles: fivethirtyeight, seaborn, seaborn-dark, etc
style.use("ggplot")

#  generate random data for learning
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y) # plot the random data points

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

# generate a new set of data to predict where they fit with our model
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis]) # predict where the new set of data fits

# plot model and predictions
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

print("Model slope:    ", model.coef_[0]) # print the slope of the model
print("Model intercept:", model.intercept_) # print the intercept of the model
