import pandas as pd
import pystruct
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_index(i, j, k, n_col):
  return 3*i*n_col + 3*j + k + 1


# USE 2.7
SIZE=2
data = pd.read_csv('data/resized_data_scaled_%d.csv'%SIZE)

print data.head(5)

col_names = data.columns
nvars = len(col_names)
n_rows = int(sqrt((nvars-1)/3.0))
n_cols = int(sqrt((nvars-1)/3.0))

print col_names
print nvars, n_rows, n_cols


# adj = np.zeros((nvars, nvars))
#
# edge_list = []
#
# print(n_rows)
# for i in range(0,(n_rows)):
#   for j in range(0,(n_cols)):
#     for k in range(0,3):
#       idx = get_index(i,j,k,n_cols)
#       edge_list.append([0, idx])
#       adj[0, idx] = 1L
#       if k==0:
#         adj[idx, get_index(i,j, 1, n_cols)] = 1L
#         adj[idx, get_index(i,j, 2, n_cols)] = 1L
#         edge_list.append([idx, get_index(i,j, 1, n_cols)])
#         edge_list.append([idx, get_index(i,j, 2, n_cols)])
#       if k==1:
#         adj[idx, get_index(i,j, 2, n_cols)] = 1L
#         edge_list.append([idx, get_index(i,j, 2, n_cols)])
#       if i+1 < n_rows:
#         adj[idx, get_index(i+1,j, k, n_cols)] = 1L
#         edge_list.append([idx, get_index(i+1,j, k, n_cols)])
#       if j+1 < n_cols:
#         adj[idx, get_index(i,j+1, k, n_cols)] = 1L
#         edge_list.append([idx, get_index(i,j+1, k, n_cols)])
#
#
# print adj
# print edge_list
#
# edge_list = np.array(edge_list).T
# print edge_list.shape

np.random.seed(42)

Y = data[['y']].values
X = data[col_names[1:]].values

print X.shape
print Y.shape

X = X.reshape(X.shape[0], SIZE, 3*SIZE, 1)

print X.shape

print X[0]

Y_tmp = np.empty((Y.shape[0], SIZE, 3*SIZE), dtype=np.uint8)
for idx, y in enumerate(Y):
    Y_tmp[idx].fill(y[0])
Y = Y_tmp

print Y.shape



# raise ValueError()


from pystruct.models import GraphCRF, GridCRF
import pystruct.learners as ssvm

# X, Y = generate_crosses_explicit(n_samples=50, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.90)

print X_train.shape

model = GridCRF(n_states=4, neighborhood=4, inference_method="max-product")
clf = ssvm.OneSlackSSVM(model=model, C=100, inference_cache=100,
                        tol=.1, max_iter=10, show_loss_every=2)
clf.fit(X_train, y_train)
Y_pred = np.array(clf.predict(X_train))

print("overall accuracy (training set): %f" % clf.score(X_train, Y_train))

# plot one example
# x, y, y_pred = X[0], Y[0], Y_pred[0]
# y_pred = y_pred.reshape(x.shape[:2])
# fig, plots = plt.subplots(1, 4, figsize=(12, 4))
# plots[0].matshow(y)
# plots[0].set_title("ground truth")
# plots[1].matshow(np.argmax(x, axis=-1))
# plots[1].set_title("input")
# plots[2].matshow(y_pred)
# plots[2].set_title("prediction")
# loss_augmented = clf.model.loss_augmented_inference(x, y, clf.w)
# loss_augmented = loss_augmented.reshape(y.shape)
# plots[3].matshow(loss_augmented)
# plots[3].set_title("loss augmented")
# for p in plots:
#     p.set_xticks(())
#     p.set_yticks(())

# # visualize weights
# w_un = clf.w[:SIZE * SIZE].reshape(SIZE, SIZE)
# # decode the symmetric pairwise potential
# w_pw = expand_sym(clf.w[SIZE * SIZE:])
#
# fig, plots = plt.subplots(1, 2, figsize=(8, 4))
# plots[0].matshow(w_un, cmap='gray', vmin=-5, vmax=5)
# plots[0].set_title("Unary weights")
# plots[1].matshow(w_pw, cmap='gray', vmin=-5, vmax=5)
# plots[1].set_title("Pairwise weights")
# for p in plots:
#     p.set_xticks(())
#     p.set_yticks(())
# plt.show()

# x_train, x_test, y_train, y_test =


#
