import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pprint import pprint
import pandas as pd
import numpy as np
import eli5

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


SIZE=2
data = pd.read_csv('data/resized_data_scaled_%d.csv'%SIZE)

pprint(data.head(5))

col_names = data.columns
nvars = len(col_names)
n_rows = int(np.sqrt((nvars-1)/3.0))
n_cols = int(np.sqrt((nvars-1)/3.0))

print(col_names)
print(f"NVARS: {nvars}, NROWS: {n_rows}, NCOLS:{n_cols}")

Y = [[str(int(y[0]))] for y in data[['y']].values.astype(np.uint8)]
X = data[col_names[1:]].values

# Sentence is image, pixel is node (or word), and RGB are features

def get_features(sample, col_names):
    features = {col_names[i+1]: float(sample[i]) for i in range(sample.shape[0])}
    features['bias'] = 1.0
    return features

print("Features:")
pprint(get_features(X[0], col_names))

X = [[get_features(X[i], col_names)] for i in range(X.shape[0])]

pprint(X[2])

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
print(len(X_train), len(y_train))
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)

print(metrics.flat_classification_report(
    y_test, y_pred, digits=3
))


from collections import Counter

# def print_transitions(trans_features):
#     for (label_from, label_to), weight in trans_features:
#         print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
#
# print("Top likely transitions:")
# print_transitions(Counter(crf.transition_features_).most_common(20))
#
# print("\nTop unlikely transitions:")
# print_transitions(Counter(crf.transition_features_).most_common()[-20:])
#
#
#
# def print_state_features(state_features):
#     for (attr, label), weight in state_features:
#         print("%0.6f %-8s %s" % (weight, label, attr))
#
# print("Top positive:")
# print_state_features(Counter(crf.state_features_).most_common(30))
#
# print("\nTop negative:")
# print_state_features(Counter(crf.state_features_).most_common()[-30:])

# eli5.show_weights(crf, top=30)
expl = eli5.explain_weights(crf, top=5)
print(eli5.format_as_text(expl))
#
