import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

def load_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

def count_colors(img):
    return len(set([tuple(img[i, j, :])
                    for i in range(img.shape[0])
                    for j in range(img.shape[1])]))


class ColorQuantization(object):

    def __init__(self, n_colors=8, algo='kmeans'):
        super(ColorQuantization, self).__init__()
        self.n_colors = n_colors
        self.algo = algo
        self._init_model(algo, n_colors)

    def _init_model(self, algo, n_colors):
        if algo == 'kmeans':
            self.model = KMeans(n_clusters=n_colors)
        elif algo == 'gmm':
            self.model = GaussianMixture(n_components=n_colors)
        else:
            raise ValueError(f"ERROR: {algo} not acceptable\nPlease choose: ['kmeans', 'gmm']")

    def _reshape(self, imgs):
        # imgs in list ( np.array(rows, cols, 3) )
        return np.concatenate([img.reshape(-1, 3) for img in imgs], axis=0)

    def fit(self, x):
        # Note do we want stack of images?
        data = self._reshape(x)
        self.model.fit(data)

    def get_centers(self):
        if self.algo == 'kmeans':
            return self.model.cluster_centers_
        return self.model.means_

    def colorize(self, imgs):
        relabeled_imgs = []
        for img in tqdm(imgs):
            rows, cols = img.shape[:2]
            img = img.reshape(-1, 3)
            labels = self.model.predict(img)
            labels = self.get_centers()[labels]
            relabeled_imgs.append(labels.reshape(rows, cols, 3).astype(int))
        return relabeled_imgs

    def plot_img(self, img, recolored_img):
        fig, ax = plt.subplots(2, 1, figsize=(8, 10),
                               subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(wspace=0.05)
        ax[0].imshow(img)
        ax[0].set_title('Original Image', size=16)
        ax[1].imshow(recolored_img)
        ax[1].set_title(f'{self.n_colors}-color Image', size=16);

    def info(self):
        return f"[INFO] ColorQuantization = \{n_colors: {self.n_colors}, algo: {self.algo} \}"


    # takes img, title, optional recolor_img, sample_size
    def plot_pixels(self, data, title, colors=None, N=10000):

        if len(data.shape) == 3:
            data = data.reshape(-1, 3) / 255
        if colors is not None and len(colors.shape) == 3:
            colors = colors.reshape(-1, 3) / 255

        i = np.random.permutation(data.shape[0])[:N]
        data = data[i]
        R = data[:, 0]
        G = data[:, 1]
        B = data[:, 2]


        c_rng = range(N)

        if colors is None:
            colors = [tuple(data[j]) for j in range(data.shape[0])]
        else:
            colors = colors[i]
        cmap = matplotlib.colors.ListedColormap(colors)

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].scatter(R, G, c=c_rng, cmap=cmap, marker='.')
        ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

        ax[1].scatter(R, B, c=c_rng, cmap=cmap, marker='.')
        ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

        fig.suptitle(title, size=20);
