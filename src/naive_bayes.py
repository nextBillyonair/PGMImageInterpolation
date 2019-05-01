from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_validate

class NaiveBayes():

    def __init__(self, type="gaussian"):
        self.scores = None
        if type == "gaussian":
            self.model = GaussianNB()
        elif type == "multinomial":
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Error: {type} is not an accepted model.")

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
