import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 1. Наївний байєсівський класифікатор
classifier_nb = GaussianNB()

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Тренування наївного байєсівського класифікатора
classifier_nb.fit(X_train, y_train)

# Прогнозування для тестових даних
y_test_pred_nb = classifier_nb.predict(X_test)

# Обчислення точності наївного байєсівського класифікатора
accuracy_nb = accuracy_score(y_test, y_test_pred_nb)
print("Accuracy of Naive Bayes classifier =", round(accuracy_nb * 100, 2), "%")

# Візуалізація роботи наївного байєсівського класифікатора
visualize_classifier(classifier_nb, X_test, y_test)

# 2. Машина опорних векторів (SVM)
classifier_svm = SVC(kernel='rbf', random_state=3)

# Тренування SVM-класифікатора
classifier_svm.fit(X_train, y_train)

# Прогнозування для тестових даних
y_test_pred_svm = classifier_svm.predict(X_test)

# Обчислення точності SVM-класифікатора
accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print("Accuracy of SVM classifier =", round(accuracy_svm * 100, 2), "%")

# Візуалізація роботи SVM-класифікатора
visualize_classifier(classifier_svm, X_test, y_test)

# Порівняння результатів
if accuracy_nb > accuracy_svm:
    print("Наївний байєсівський класифікатор показав кращу точність.")
elif accuracy_nb == accuracy_svm:
    print("Однакові результати")
else:
    print("SVM класифікатор показав кращу точність.")

num_folds = 2
accuracy_values = cross_val_score(classifier_nb, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
precision_values = cross_val_score(classifier_nb, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = cross_val_score(classifier_nb, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = cross_val_score(classifier_nb, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

num_folds = 2
accuracy_values = cross_val_score(classifier_svm, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
precision_values = cross_val_score(classifier_svm, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
recall_values = cross_val_score(classifier_svm, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
f1_values = cross_val_score(classifier_svm, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")
