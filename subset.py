import numpy as np
import cv2
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import ntpath

import stats


data_path = "../PGMData/jpg2/"
xml_path = "../PGMData/xml2/"
img_path = "../PGMData/imgs/"

top1 = "foto"
top2 = "schilderij"

def move_imgs(xmls):
    top1count = 0
    top2count = 0
    for xml_file in tqdm(xmls):
        xml = stats.read_xml(xml_file)
        creator = stats.extract_type(xml)
        if creator in [top1, top2]:
            file = ntpath.basename(xml_file).strip('.xml')
            img = cv2.imread(f"{data_path}{file}.jpg")
            base_name = None
            if creator == top1:
                base_name = f"foto_{top1count}.jpg"
                top1count += 1
            else:
                base_name = f"schilderij_{top2count}.jpg"
                top2count += 1
            cv2.imwrite(f"{img_path}{base_name}", img)
            # img = cv2.imread('')



if __name__ == '__main__':
    # imgs = get_img_files()
    xmls = stats.get_xml_files()
    move_imgs(xmls)
