import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# Plot size to 14" x 7"
matplotlib.rc('figure', figsize = (14, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Do not display top and right frame lines
matplotlib.rc('axes.spines', top = False, right = False)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')

# colors = {0:'#539caf', 1:'#7663b0'}
colors = { 0:"#003f5c", 1:"#444e86", 2:"#955196", 3:"#dd5182", 4:"#ff6e54", 5:"#ffa600"}

PATH = "../../PGMData/imgs/"
LIMIT = 2269
TYPES = ['schilderij', 'foto', 'tekening', 'prent']
ENG_TYPES = ["Paintings", "Photos", "Drawings", "Prints"]

def count_colors(img):
    return len(set([tuple(img[i, j, :])
                    for i in range(img.shape[0])
                    for j in range(img.shape[1])]))

def load_img(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

def load_type(type):
    return [load_img(f"{PATH}{type}_{i}.jpg") for i in tqdm(range(LIMIT))]

def load_dataset():
    return {type: load_type(type) for type in TYPES}

def downsize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).reshape(-1)

# batch
def downsize_set(data, size):
    return {label: [downsize(i, size)
            for i in data[label]] for label in data}

def plot_img(img, size = None):
    if len(img.shape) == 1:
        img = img.reshape(size, size, 3)
    plt.axis('off')
    plt.imshow(img)

def plot_pixels(data, title, colors=None, N=10000, s=4, alpha=0.4):
    if len(data.shape) == 3:
        data = data.reshape(-1, 3) / 255
    elif len(data.shape) == 1:
        data = data / 255.0
        data = data.reshape(-1, 3)

    if colors is not None and len(colors.shape) == 3:
        colors = colors.reshape(-1, 3) / 255

    i = np.random.permutation(data.shape[0])[:N]
    data = data[i]
    R = data[:, 0]
    G = data[:, 1]
    B = data[:, 2]

    c_rng = range(N if data.shape[0] >= N else data.shape[0])

    if colors is None:
        colors = [tuple(data[j]) for j in range(data.shape[0])]
    else:
        colors = colors[i]
    cmap = matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, c=c_rng, cmap=cmap, marker='.', s=s, alpha=alpha)
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, c=c_rng, cmap=cmap, marker='.', s=s, alpha=alpha)
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);

def stack(imgs):
    (l1, d1), (l2, d2), (l3, d3), (l4, d4) = imgs.items()
    train = np.stack(tuple(d1 + d2 + d3 + d4), axis=0)
    test = [0 for _ in range(len(d1))] + \
           [1 for _ in range(len(d2))] + \
           [2 for _ in range(len(d3))] + \
           [3 for _ in range(len(d4))]
    return train, test

def make_train_test_sets(imgs, p=0.1):
    train, test = stack(imgs)
    return train_test_split(train, test, test_size=p), {0:l1, 1:l2, 2:l3, 3:l4}

def make_subplot(rows, cols):
    fig, axs = plt.subplots(rows, cols)
    return fig, axs

def lineplot_ci(ax, sizes, mean, std, label, x_label, y_label, title, color=0):
    color = colors[color]
    Lower, Upper = get_ci(mean, std)
    ax.plot(sizes, mean, lw=1, label=label, color=color, marker='o', markersize=4)
    ax.fill_between(sizes, Upper, Lower, alpha=0.4, label="95% CI", color=color)
    ax.set_title(title)
    ax.legend(loc = 'lower right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def lineplot(ax, x_data, y_data, x_label, y_label, title, label, color=0):
    color = colors[color]
    ax.plot(x_data, y_data, lw = 2, label=label, color = color, alpha = 1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def get_ci(means, stds):
    Lower = [means[i] - (1.96)*(stds[i]) for i in range(len(means))]
    Upper = [means[i] + (1.96)*(stds[i]) for i in range(len(means))]
    return Lower, Upper







# EOF
