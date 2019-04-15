import numpy as np
import cv2
import glob
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm
import pandas as pd
from pprint import pprint


data_path = "../PGMData/jpg2/"
xml_path = "../PGMData/xml2/"

def get_files(path, suffix, n=None):
    return glob.glob(f"{path}*.{suffix}")[:n]

def get_img_files(n=None):
    return get_files(data_path, 'jpg', n)

def get_xml_files(n=None):
    return get_files(xml_path, 'xml', n)

def read_xml(file):
    with open(file) as fp:
        return BeautifulSoup(fp, "xml")

def extract_creator(xml):
    if xml.creator is None: return None
    return xml.creator.string.split(":")[1][1:]

def extract_type(xml):
    if xml.type is None: return None
    return xml.type.string

def to_pandas(counter):
    return pd.DataFrame.from_dict(counter, orient='index', columns=['Count'])

def count_artists(xml_files):
    counter = Counter()
    for xml_file in tqdm(xmls):
        xml = read_xml(xml_file)
        creator = extract_creator(xml)
        if creator is not None:
            counter[creator] += 1
    return counter

def count_type(xml_files):
    counter = Counter()
    for xml_file in tqdm(xmls):
        xml = read_xml(xml_file)
        creator = extract_type(xml)
        if creator is not None:
            counter[creator] += 1
    return counter

def save(df, out_path='freq.csv'):
    df.to_csv(out_path, sep="|")

if __name__ == '__main__':
    xmls = get_xml_files()
    # counts = count_artists(xmls)
    counts = count_type(xmls)
    pprint(counts.most_common(30))
    df = to_pandas(counts)
    df.sort_values(by="Count", ascending=False, inplace=True)
    save(df, out_path="freq_type.csv")
