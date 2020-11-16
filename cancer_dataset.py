# Objective: Predict if a person is bengin or malignant for breast cancer.

# -----------------------------------------------------------------------------------

# ! IMPORT MODULES AND PREPARE DATASET

# Import sklearn modules.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Perceptron

# Import other modules.
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from os import path, getcwd

print(getcwd())

dataset = pd.read_csv(path.join('Projects/Cancer Dataset', 'data.csv'))
data = dataset.loc[:, 'radius_mean':'fractal_dimension_worst'].to_numpy()
target = pd.get_dummies(dataset.loc[:, 'diagnosis']).loc[:, "B"].to_numpy()

# NOTE: If sample has cancer, sample=0, else sample=1.

# Split data
print(np.bincount(target))
X_train, X_test, y_train, y_test = train_test_split(
    data, target,
    stratify=target, random_state=42)

print(np.bincount(y_train))
print(np.bincount(y_test))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# ---------------------------------------------------------------------------------------

# ! CHECK IF THE DATA IS LINEAR THROUGH GRAPHING AND A LINEAR MODEL

tsne = TSNE(random_state=42)
data_trans = tsne.fit_transform(data)

plt.scatter(data_trans[:, 0], data_trans[:, 1], c=target)

# Apply linear model
lin_clf = Perceptron(random_state=42)

lin_clf.fit(X_train, y_train)

print(lin_clf.score(X_train, y_train))
print(lin_clf.score(X_test, y_test))

# TODO: Print results of linear model.
