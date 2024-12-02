import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from LR_4_task_6 import plot_learning_curves

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)
print(X)
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]
# Створення об'єкта лінійного регресора
linear_regressor = linear_model.LinearRegression()
# plot_learning_curves(linear_regressor, X, y)
linear_regressor.fit(X_train, y_train)
# Прогнозування результату
y_test_pred = linear_regressor.predict(X_test)

# %%
print("Linear regressor performance:")
print("Mean absolute error =",
round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =",
round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =",
round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =",
round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# %%
# Побудова графіка
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# %%
# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

# %%
model = linear_model.LinearRegression()
model.fit(X_train_transformed, y_train)

X_plot = np.linspace(min(X_train), max(X_train), 500).reshape(-1, 1)  # Генеруємо точки для побудови кривої
X_plot_transformed = polynomial.transform(X_plot)  # Трансформуємо точки у поліноміальну форму
y_plot = model.predict(X_plot_transformed)  # Передбачення моделі


# %%
plt.scatter(X_train, y_train, color="blue", label="Фактичні дані")  # Фактичні дані
plt.plot(X_plot, y_plot, color="red", label="Поліноміальна регресія")  # Крива моделі
plt.xlabel("Значення X")
plt.ylabel("Значення Y")
plt.title("Поліноміальна регресія (degree=10)")
plt.legend()
plt.show()

# %%
print("Regression Coeficient")
print(model.coef_)
print("Regression Interceptor")
print(model.intercept_)


