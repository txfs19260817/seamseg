'''
RENAME FILENAMES UNDER THE DATASET
ACCORDING TO THE CITYSCAPES MANNER
'''

import argparse
import glob
import json
import shutil
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir
from collections import namedtuple

import numpy as np
import tqdm
import umsgpack
from PIL import Image
from pycococreatortools import pycococreatortools as pct

parser = argparse.ArgumentParser(description="Convert iSAID to seamseg format")
parser.add_argument(
    "root_dir",
    metavar="ROOT_DIR",
    type=str,
    default="iSAID_patches",
    help="Root directory of iSAID"
)
parser.add_argument(
    "out_dir",
    metavar="OUT_DIR",
    type=str,
    default="iSAID_seamseg",
    help="Output directory"
)

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    'm_color'       , # The color of this label
])

cs_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
    # Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) , 0      ), # background
    Label(  'background'           ,  0 ,       15 , 'void'            , 0       , False        , False        , (  0,  0,  0) , 0      ),
    Label(  'ship'                 ,  1 ,        0 , 'transport'       , 1       , True         , False        , (  0,  0, 63) , 4128768),
    Label(  'storage_tank'         ,  2 ,        1 , 'transport'       , 1       , True         , False        , (  0, 63, 63) , 4144896),
    Label(  'baseball_diamond'     ,  3 ,        2 , 'land'            , 2       , True         , False        , (  0, 63,  0) , 16128  ),
    Label(  'tennis_court'         ,  4 ,        3 , 'land'            , 2       , True         , False        , (  0, 63,127) , 8339200),
    Label(  'basketball_court'     ,  5 ,        4 , 'land'            , 2       , True         , False        , (  0, 63,191) , 12533504),
    Label(  'Ground_Track_Field'   ,  6 ,        5 , 'land'            , 2       , True         , False        , (  0, 63,255) , 16727808),
    Label(  'Bridge'               ,  7 ,        6 , 'land'            , 2       , True         , False        , (  0,127, 63) , 4161280),
    Label(  'Large_Vehicle'        ,  8 ,        7 , 'transport'       , 1       , True         , False        , (  0,127,127) , 8355584),
    Label(  'Small_Vehicle'        ,  9 ,        8 , 'transport'       , 1       , True         , False        , (  0,  0,127) , 8323072),
    Label(  'Helicopter'           , 10 ,        9 , 'transport'       , 1       , True         , False        , (  0,  0,191) , 12517376),
    Label(  'Swimming_pool'        , 11 ,       10 , 'land'            , 2       , True         , False        , (  0,  0,255) , 16711680),
    Label(  'Roundabout'           , 12 ,       11 , 'land'            , 2       , True         , False        , (  0,191,127) , 8371968),
    Label(  'Soccer_ball_field'    , 13 ,       12 , 'land'            , 2       , True         , False        , (  0,127,191) , 12549888),
    Label(  'plane'                , 14 ,       13 , 'transport'       , 1       , True         , False        , (  0,127,255) , 16744192),
    Label(  'Harbor'               , 15 ,       14 , 'transport'       , 1       , True         , False        , (  0,100,155) , 10183680),
]

_SPLITS = {
    "train": ("images/train", "gtFine/train"),
    "val": ("images/val", "gtFine/val"),
}
_INSTANCE_EXT = "_instanceIds.png"
_IMAGE_EXT = "_leftImg8bit.png"


def main(args):
    print("Loading iSAID from", args.root_dir)
    num_stuff, num_thing = _get_meta()

    # Prepare directories
    img_dir = path.join(args.out_dir, "img")
    _ensure_dir(img_dir)
    msk_dir = path.join(args.out_dir, "msk")
    _ensure_dir(msk_dir)
    lst_dir = path.join(args.out_dir, "lst")
    _ensure_dir(lst_dir)
    coco_dir = path.join(args.out_dir, "coco")
    _ensure_dir(coco_dir)

    # COCO-style category list
    coco_categories = []
    for lbl in cs_labels:
        if lbl.trainId != 255 and lbl.trainId != -1 and lbl.hasInstances:
            coco_categories.append({
                "id": lbl.trainId,
                "name": lbl.name
            })

    # Process splits
    images = []
    for split, (split_img_subdir, split_msk_subdir) in _SPLITS.items():
        print("Converting", split, "...")

        img_base_dir = path.join(args.root_dir, split_img_subdir)
        msk_base_dir = path.join(args.root_dir, split_msk_subdir)
        img_list = _get_images(msk_base_dir)

        # Write the list file
        with open(path.join(lst_dir, split + ".txt"), "w") as fid:
            fid.writelines(img_id + "\n" for _, img_id, _ in img_list)

        # Convert to COCO detection format
        coco_out = {
            "info": {"version": "1.0"},
            "images": [],
            "categories": coco_categories,
            "annotations": []
        }

        # Process images in parallel
        worker = _Worker(img_base_dir, msk_base_dir, img_dir, msk_dir)
        with Pool(initializer=_init_counter, initargs=(_Counter(0),)) as pool:
            total = len(img_list)
            for img_meta, coco_img, coco_ann in tqdm.tqdm(pool.imap(worker, img_list, 8), total=total):
                images.append(img_meta)

                # COCO annotation
                coco_out["images"].append(coco_img)
                coco_out["annotations"] += coco_ann

        # Write COCO detection format annotation
        with open(path.join(coco_dir, split + ".json"), "w") as fid:
            json.dump(coco_out, fid)

    # Write meta-data
    print("Writing meta-data")
    meta = {
        "images": images,
        "meta": {
            "num_stuff": num_stuff,
            "num_thing": num_thing,
            "categories": [],
            "palette": [],
            "original_ids": []
        }
    }

    for lbl in cs_labels:
        if lbl.trainId != 255 and lbl.trainId != -1:
            meta["meta"]["categories"].append(lbl.name)
            meta["meta"]["palette"].append(lbl.color)
            meta["meta"]["original_ids"].append(lbl.id)

    with open(path.join(args.out_dir, "metadata.bin"), "wb") as fid:
        umsgpack.dump(meta, fid, encoding="utf-8")


def _get_images(base_dir):
    img_list = []
    for subdir in listdir(base_dir):
        subdir_abs = path.join(base_dir, subdir)
        if path.isdir(subdir_abs):
            for img in glob.glob(path.join(subdir_abs, "*" + _INSTANCE_EXT)):
                _, img = path.split(img)

                parts = img.split("_")
                img_id = "_".join(parts[:-2])
                lbl_cat = parts[-2]

                img_list.append((subdir, img_id, lbl_cat))

    return img_list


def _get_meta():
    num_stuff = sum(1 for lbl in cs_labels if 0 <= lbl.trainId < 255 and not lbl.hasInstances)
    num_thing = sum(1 for lbl in cs_labels if 0 <= lbl.trainId < 255 and lbl.hasInstances)
    return num_stuff, num_thing


def _ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


class _Worker:
    def __init__(self, img_base_dir, msk_base_dir, img_dir, msk_dir):
        self.img_base_dir = img_base_dir
        self.msk_base_dir = msk_base_dir
        self.img_dir = img_dir
        self.msk_dir = msk_dir

    def __call__(self, img_desc):
        img_dir, img_id, lbl_cat = img_desc
        coco_ann = []

        # Load the annotation
        with Image.open(path.join(self.msk_base_dir, img_dir, img_id + "_" + lbl_cat + _INSTANCE_EXT)) as lbl_img:
            lbl = np.array(lbl_img)
            lbl_size = lbl_img.size

        ids = np.unique(lbl)

        # Compress the labels and compute cat
        lbl_out = np.zeros(lbl.shape, np.int32)
        cat = [255]
        iscrowd = [0]
        for city_id in ids:
            if city_id < 1000:
                # Stuff or group
                cls_i = city_id
                iscrowd_i = cs_labels[cls_i].hasInstances
            else:
                # Instance
                cls_i = city_id // 1000
                iscrowd_i = False

            # If it's a void class just skip it
            if cs_labels[cls_i].trainId == 255 or cs_labels[cls_i].trainId == -1:
                continue

            # Extract all necessary information
            iss_class_id = cs_labels[cls_i].trainId
            iss_instance_id = len(cat)
            mask_i = lbl == city_id

            # Save ISS format annotation
            cat.append(iss_class_id)
            iscrowd.append(1 if iscrowd_i else 0)
            lbl_out[mask_i] = iss_instance_id

            # Compute COCO detection format annotation
            if cs_labels[cls_i].hasInstances:
                category_info = {"id": iss_class_id, "is_crowd": iscrowd_i}
                coco_ann_i = pct.create_annotation_info(
                    counter.increment(), img_id, category_info, mask_i, lbl_size, tolerance=2)
                if coco_ann_i is not None:
                    coco_ann.append(coco_ann_i)

        # COCO detection format image annotation
        coco_img = pct.create_image_info(img_id, path.join(img_dir, img_id + _IMAGE_EXT), lbl_size)

        # Write output
        Image.fromarray(lbl_out).save(path.join(self.msk_dir, img_id + ".png"))
        shutil.copy(path.join(self.img_base_dir, img_dir, img_id + _IMAGE_EXT),
                    path.join(self.img_dir, img_id + ".png"))

        img_meta = {
            "id": img_id,
            "cat": cat,
            "size": (lbl_size[1], lbl_size[0]),
            "iscrowd": iscrowd
        }

        return img_meta, coco_img, coco_ann


def _init_counter(c):
    global counter
    counter = c


class _Counter:
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            val = self.val.value
            self.val.value += 1
        return val


if __name__ == "__main__":
    main(parser.parse_args())
