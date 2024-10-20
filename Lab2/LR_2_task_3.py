from sklearn.datasets import load_iris
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

######################################################## КРОК 1 ####################################################
iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей:{}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data:{}".format(iris_dataset['data'].shape))

iris_df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)

# Виведення значень ознак для перших п'яти прикладів
print(iris_df.head())

print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))

######################################################## КРОК 2 ####################################################
# Діаграма розмаху
iris_df.plot(kind='box', subplots=True, layout=(2, 2),
             sharex=False, sharey=False)
pyplot.show()

# Гістограма розподілу атрибутів датасета
iris_df.hist()
pyplot.show()

# Матриця діаграм розсіювання
pd.plotting.scatter_matrix(iris_df)
pyplot.show()

######################################################## КРОК 3 ####################################################
iris_df['target'] = iris_dataset.target  # Додаємо мітки як окремий стовпець
# Розділення датасету на навчальну та контрольну вибірки
array = iris_df.values
# Вибір перших 4-х стовпців
X = array[:, 0:4]
# Вибір 5-го стовпця
y = array[:, 4]
# Розділення X и y на навчальну и контрольну вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

######################################################## КРОК 4 ####################################################

# Завантажуємо алгоритми моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# оцінюємо модель на кожній ітерації
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

######################################################## КРОК 6 ####################################################
# Створюємо прогноз на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

######################################################## КРОК 7 ####################################################
# Оцінюємо прогноз
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

######################################################## КРОК 8 ####################################################
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма масиву X_new: {}".format(X_new.shape))

# prediction = KNeighborsClassifier().fit(X_train, Y_train).predict(X_new) # код із моделлю із прикладу KNN
prediction = SVC(gamma='auto').fit(X_train, Y_train).predict(X_new) # код із моделлю SVC


print("Прогноз: {}".format(prediction))

predicted_class = iris_dataset['target_names'][int(prediction[0])]
print("Спрогнозований клас: {}".format(predicted_class))
