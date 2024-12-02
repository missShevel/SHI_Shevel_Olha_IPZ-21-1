
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def plot_learning_curves (model, X, y): 
    X_train, X_val, y_train, y_val= train_test_split (X, y, test_size=0.2)
    train_errors, val_errors = [], [] 
    for m in range (1, len (X_train)):
        model.fit (X_train [:m], y_train [:m]) 
        y_train_predict = model.predict (X_train[:m])
        y_val_predict = model.predict (X_val)
        train_errors.append(mean_squared_error (y_train_predict,y_train[:m]))
        val_errors.append (mean_squared_error (y_val_predict, y_val)) 
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]
# Створення об'єкта лінійного регресора
linear_regressor = linear_model.LinearRegression()
plot_learning_curves(linear_regressor, X, y)


polynomial_regression10 = Pipeline([("poly_features", PolynomialFeatures(degree=2, include_bias=False)), ("lin_reg", linear_model.LinearRegression())])
plot_learning_curves(polynomial_regression10, X, y)
polynomial_regression2 = Pipeline([("poly_features", PolynomialFeatures(degree=2, include_bias=False)), ("lin_reg", linear_model.LinearRegression())])
plot_learning_curves(polynomial_regression2, X, y)
