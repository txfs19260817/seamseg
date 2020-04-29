'''
Convert train id to id
on images predicted by script
"test_panoptic_single.py"
'''


import argparse
import multiprocessing
import os
import numpy as np
from PIL import Image
from cityscapesscripts.helpers.labels import labels as cs_labels
from tqdm import tqdm


## args

parser = argparse.ArgumentParser(description="Panoptic testing script")
parser.add_argument(
    "--inputs",
    type=str,
    default="cityscapes_weakly_1to2_trainId",
    help="images predicted by script 'test_panoptic_single.py'"
)
parser.add_argument(
    "--outputs",
    type=str,
    default="cityscapes_weakly_1to2_id",
    help="converted images output directory"
)
args = parser.parse_args()

## const

# paths
inputs = args.inputs
outputs = args.outputs
train_citys = [
    'aachen',
    'bochum',
    'bremen',
    'cologne',
    'darmstadt',
    'dusseldorf',
    'erfurt',
    'hamburg',
    'hanover',
    'jena',
    'krefeld',
    'monchengladbach',
    'strasbourg',
    'stuttgart',
    'tubingen',
    'ulm',
    'weimar',
    'zurich'
]

# trainId to id
t2i = {}
for c in cs_labels:
    t2i[c.trainId] = c.id
t2i[255] = 0


def worker(f):
    img = Image.open(os.path.join(inputs, f))
    img = np.asarray(img).copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 1000:
                cata, inst_id = img[i][j] // 1000, img[i][j] % 1000
                cata = t2i[cata]
                img[i][j] = int(cata * 1000 + inst_id)
            else:
                img[i][j] = t2i[img[i][j]]
    city = f.split('_')[0]
    Image.fromarray(img).save(os.path.join(outputs, city, f))


def main():
    # mkdir
    for i in train_citys:
        try:
            os.makedirs(os.path.join(outputs, i))
        except:
            pass
    
    # multiprocessing
    p = multiprocessing.Pool()
    files = os.listdir(inputs)
    p.map(worker, files)
    p.close()
    p.join()


if __name__ == "__main__":
    main()
    print('Done')