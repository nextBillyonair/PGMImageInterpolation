from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_validate

class LogisticRegression():

    def __init__(self):
        self.model = LR(solver='lbfgs', multi_class='multinomial', n_jobs=-1)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, data, labels, cv=3):
        self.metrics = cross_validate(self.model, data, labels, cv=cv,
                                      n_jobs=-1, return_train_score=True)

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
