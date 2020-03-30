import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

DATASET_ROOT = "datasets/obj365/"
TRAIN_JSON = "datasets/obj365/objects365_Tiny_train.json"
VAL_JSON = "datasets/obj365/objects365_Tiny_val.json"

TRAIN_PATH = "datasets/obj365/train/"
VAL_PATH = "datasets/obj365/val/"

PREDEFINED_SPLITS_DATASET = {
    "obj365_train": (TRAIN_PATH, TRAIN_JSON),
    "obj365_val": (VAL_PATH, VAL_JSON)
}

rootdir = "/home/houbowei/adet/"
obj = os.path.join(rootdir, "./datasets/obj365/objects365_Tiny_train.json")
with open(obj, 'r') as obj:
    obj = json.load(obj)
    things = [0]*65
for cat in obj['categories']:
    things[cat['id']-301] = cat['name']

thing_dataset_id_to_contiguous_id = {i+301: i for i in range(65)}

DatasetCatalog.register(
    "obj365_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "coco"))
DatasetCatalog.register(
    "obj365_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco"))

MetadataCatalog.get('obj365_val').set(evaluator_type='coco',
                                      json_file=VAL_JSON,
                                      image_root=VAL_PATH,
                                      thing_classes=things,
                                      thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id
                                      )
MetadataCatalog.get('obj365_train').set(
    evaluator_type='coco',
    json_file=TRAIN_JSON,
    image_root=TRAIN_PATH,
    thing_classes=things,
    thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
