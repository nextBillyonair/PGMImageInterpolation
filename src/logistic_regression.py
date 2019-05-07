from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import time

class LogisticRegression():

    def __init__(self):
        self.model = LR(solver='lbfgs', multi_class='multinomial')

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, X, Y, cv=3):
        self.metrics = {'fit_time':[], 'score_time':[], 'train_score':[], 'test_score':[]}
        for i in range(cv):
            self.model = LR(solver='lbfgs', multi_class='multinomial')
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
            start = time.time()
            self.model.fit(X_train, y_train)
            self.metrics['fit_time'].append(time.time() - start)
            self.metrics['train_score'].append(self.model.score(X_train, y_train))
            start = time.time()
            self.metrics['test_score'].append(self.model.score(X_test, y_test))
            self.metrics['score_time'].append(time.time() - start)
        # self.metrics = cross_validate(self.model, data, labels, cv=cv,
        #                               n_jobs=-1, return_train_score=True)

    def score_by_class(self, x_test, y_test):
        class_correct = list(0. for i in range(4))
        class_total = list(0. for i in range(4))
        predicted = self.predict(x_test)
        c = (predicted == y_test)
        for i in range(x_test.shape[0]):
            label = y_test[i]
            class_correct[label] += c[i]
            class_total[label] += 1

        class_accuracy = [class_correct[label] / class_total[label]
                          for label in range(4)]

        return class_accuracy, class_correct, class_total
