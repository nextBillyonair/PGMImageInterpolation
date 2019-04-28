import numpy as np
import cv2
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import ntpath
import random

import stats


data_path = "../PGMData/jpg2/"
xml_path = "../PGMData/xml2/"
img_path = "../PGMData/imgs/"

top1 = "tekening"
top2 = "foto"
top3 = "schilderij"
top4 = "prent"

limit = 2269

def move_imgs(xmls):
    top1count = 0
    top2count = 0
    top3count = 0
    top4count = 0
    for xml_file in tqdm(xmls):
        xml = stats.read_xml(xml_file)
        creator = stats.extract_type(xml)
        if creator in [top1, top2, top3, top4]:
            file = ntpath.basename(xml_file).strip('.xml')
            img = cv2.imread(f"{data_path}{file}.jpg")
            base_name = None
            if creator == top1 and top1count < limit:
                base_name = f"tekening_{top1count}.jpg"
                top1count += 1
            elif creator == top2 and top2count < limit:
                base_name = f"foto_{top2count}.jpg"
                top2count += 1
            elif creator == top3 and top3count < limit:
                base_name = f"schilderij_{top3count}.jpg"
                top3count += 1
            elif creator == top4 and top4count < limit:
                base_name = f"prent_{top4count}.jpg"
                top4count += 1
            if base_name is not None:
                cv2.imwrite(f"{img_path}{base_name}", img)
            # img = cv2.imread('')



if __name__ == '__main__':
    # imgs = get_img_files()
    xmls = stats.get_xml_files()
    random.shuffle(xmls)
    move_imgs(xmls)
