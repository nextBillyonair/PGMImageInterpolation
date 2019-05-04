# Defining the Bayesian Model
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import time
from tqdm import tqdm
import img_utils


class GBN(BaseEstimator):

    def __init__(self, size):
        self.size = size
        self.num_pixels = size**2
        self.num_features = 3*self.num_pixels
        self.columns = ['y'] + [f'X_{i}|{j}|{k}' for i in range(size)
                                for j in range(size) for k in range(3)]
        # print(self.columns)
        self.edges = [(x_node, 'y') for x_node in self.columns[1:]]
        self.model = BayesianModel(self.edges)


    def fit(self, X):
        # X, y = check_X_y(X, y)
        # self.classes_ = unique_labels(y)
        # # do color quant
        # data = np.hstack((np.array(y).reshape(-1, 1), X))
        # data = pd.DataFrame(data, columns=self.columns)
        # print(data)
        for node in tqdm(self.columns):
            MaximumLikelihoodEstimator(self.model, data).estimate_cpd(node)
        start_time = time.time()
        # self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        self.fit_time = time.time() - start_time
        print(self.fit_time)


    def predict(self, X):
        check_is_fitted(self, ['classes_'])
        X = check_array(X)
        start_time = time.time()
        # predict, or marginalize
        self.score_time = time.time() - start_time



# EOF

if __name__ == '__main__':
    SIZE=1
    model = GBN(SIZE)
    data = pd.read_csv(f'data/resized_data_{SIZE}.csv')
    # imgs = img_utils.load_dataset()
    # data, labels = img_utils.stack(img_utils.downsize_set(imgs, SIZE))
    model.fit(data)
    for cpd in model.model.get_cpds():
        print("CPD of {variable}:".format(variable=cpd.variable))
        print(cpd)
