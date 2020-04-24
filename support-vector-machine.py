import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(font_scale=1.2)
import pickle  # for serialization

# Read in muffin and cupcake ingredient data
recipes = pd.read_csv('data/recipes_muffins_cupcakes.csv')
recipes.head()

# Plot two ingredients
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})

# Specify inputs for the model
ingredients = recipes[['Flour', 'Sugar']].values # we'll use only 2 of the many so it's easier to visualize
type_label = np.where(recipes['Type'] == 'Muffin', 0, 1) # type is 0 where it's Muffin 1 otherwise
recipe_features = recipes.columns.values[1:].tolist() # Feature names / columns

# Fit the SVM model
# there are many different estimators:
#     Regression: linear SV classification LinearSVC, linear SV regression Linear SVR
#     Classification: SVC
#     Outlier detector: One Class SVM
model = svm.SVC(kernel='linear')
# in more complex examples I should use validation set, not just training set
model.fit(ingredients, type_label)
print(model) # if I print the model it will show the defaults used to create the model


# Visualize the results

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1] # this is the slope
xx = np.linspace(30, 60) # create a linearly distributed array of values between 30 and 60
yy = a * xx - (model.intercept_[0]) / w[1] # this is the hypothesis, the equasion of a line

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0]) # lower line (goes through the nearest muffin)
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0]) # upper line (goes through the nearest cupcake)

# Plot the hyperplane
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')

# Look at the margins and support vectors
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80, facecolors='none')


# Predict new cases

# Create a function to guess when a recipe is a muffin or a cupcake
def muffin_or_cupcake(flour, sugar):
    if (model.predict([[flour, sugar]])) == 0:
        print('You\'re looking at a muffin recipe!')
    else:
        print('You\'re looking at a cupcake recipe!')


# Predict if 50 parts flour and 20 parts sugar
muffin_or_cupcake(50, 20)

# Plot the point to visually see where the point lies
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50, 20, 'yo', markersize='9')

# Predict if 40 parts flour and 20 parts sugar
muffin_or_cupcake(40, 20)

muffin_cupcake_dict = {'muffin_cupcake_model': model, 'muffin_cupcake_features': ['Flour', 'Sugar'],
                       'all_features': recipe_features}
muffin_cupcake_dict

# Serialize the model and write it to a file
pickle.dump(muffin_cupcake_dict, open("data/muffin_cupcake_dict.p", "wb"))

# S = String
pickle.dumps(muffin_cupcake_dict)
