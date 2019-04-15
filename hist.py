import numpy as np
import cv2
import glob
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint

import stats

img_path = "../PGMData/imgs/"

def get_hist(filename, colors=3):
    if colors == 1:
        img = cv2.imread(filename, 0)
    else:
        img = cv2.imread(filename)
    return [cv2.calcHist([img], [i], None, [256], [0,256]) for i in range(colors)]

def plot(hists, colors=3, c='r'):
    if colors == 3:
        color = ('b','g','r')
    else:
        color = (c)
    for i, col in enumerate(color):
        plt.plot(hists[i], color = col)
        plt.xlim([0,256])

def main1(gray=True):
    files = stats.get_files(img_path, 'jpg')
    colors = 1 if gray else 3
    global_hist = [np.zeros((256,1)) for i in range(colors)]
    for file in tqdm(files):
        hists = get_hist(file, colors=colors)
        for i, hist in enumerate(hists):
            global_hist[i] += hist

    plot(global_hist, colors)
    plt.show()

def main2(gray=True):
    files = stats.get_files(img_path, 'jpg')
    colors = 1 if gray else 3
    global_hist = [[np.zeros((256,1)) for i in range(colors)] for j in range(2)]
    for file in tqdm(files):
        hists = get_hist(file, colors=colors)
        index = 0
        if "foto" in file:
            index = 1
        for i, hist in enumerate(hists):
            global_hist[index][i] += hist

    plot(global_hist[0], colors, c='b')
    plot(global_hist[1], colors, c='r')
    plt.show()

if __name__ == '__main__':
    main2(True)
    main2(False)
    main1(False)
