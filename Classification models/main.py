from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression
from sklearn.svm import SVC
import data_preprocessing as data_pre
import test_data


# Taking data from data_preprocessing.py file
x_train2, x_test2, y_train2, y_test2 = data_pre.X_train, data_pre.X_test, data_pre.y_train, data_pre.y_test
x_train_vect2, x_test_vect2 = data_pre.X_train_vect, data_pre.X_test_vect

# Taking data from test_data.py file which contains data from test folder
final_input_test2 = test_data.final_input_test
final_input_test_vect2 = test_data.final_input_test_vect

# Naive Bayes model
naive_model = NaiveBayes()
naive_model.fit(x_train2, y_train2)

naive_test_predict = naive_model.predict(x_test2)
naive_train_predict = naive_model.predict(x_train2)

# Predictions for data from test folder
final_naive_predict = naive_model.predict(final_input_test2)

naive_test_accuracy = sum(naive_test_predict == y_test2) / len(naive_test_predict)
naive_train_accuracy = sum(naive_train_predict == y_train2) / len(naive_train_predict)

print(f'Accuracy of Naive Bayes on Training data found from internet is: {naive_train_accuracy * 100:.2f}%')
print(f'Accuracy of Naive Bayes on Testing data found from internet is: {naive_test_accuracy * 100:.2f}%')

print('Predictions of data from test folder for Naive Bayes are:')
print(final_naive_predict)

# Logistic Regression
logReg_model = LogisticRegression()
logReg_model.fit(x_train_vect2, y_train2)

logReg_test_predict = logReg_model.predict(x_test_vect2)
logReg_train_predict = logReg_model.predict(x_train_vect2)

final_logReg_predict = logReg_model.predict(final_input_test_vect2)

logReg_test_accuracy = sum(logReg_test_predict == y_test2) / len(logReg_test_predict)
logReg_train_accuracy = sum(logReg_train_predict == y_train2) / len(logReg_train_predict)

print(f'Accuracy of Logistic Regression on Training data found from internet is: {logReg_train_accuracy * 100:.2f}%')
print(f'Accuracy of Logistic Regression on Testing data found from internet is: {logReg_test_accuracy * 100:.2f}%')

print('Predictions of data from test folder for Logistic Regression are:')
print(final_logReg_predict)

# SVM
svm_model = SVC()
svm_model.fit(x_train_vect2, y_train2)

svm_test_predict = svm_model.predict(x_test_vect2)
svm_train_predict = svm_model.predict(x_train_vect2)

final_svm_predict = svm_model.predict(final_input_test_vect2)

svm_train_accuracy = sum(svm_train_predict == y_train2) / len(svm_train_predict)
svm_test_accuracy = sum(svm_test_predict == y_test2) / len(svm_test_predict)

print(f'Accuracy of SVM on Training data found from internet is: {svm_train_accuracy * 100:.2f}%')
print(f'Accuracy of SVM on Testing data found from internet is: {svm_test_accuracy * 100:.2f}%')

print('Predictions of data from test folder for SVM are:')
print(final_svm_predict)