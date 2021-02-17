# %%
import torch, torchvision

# import some common libraries
import numpy as np
#import cv2
import random
import os

import detectron2
from detectron2.utils.logger import setup_logger
#%matplotlib notebook
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Trainer and Configuration
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

detectron2.__version__

# %%
# General definitions:
mask_it = True
train = False

# Register datasets
path = '../'
pic_path = ''

register_coco_instances(
    "my_dataset_train", {}, os.path.join(path, "train.json"), pic_path
)
register_coco_instances(
    "my_dataset_valid", {}, os.path.join(path, "valid.json"), pic_path
)
register_coco_instances(
    "my_dataset_test", {}, os.path.join(path, "test.json"), pic_path
)
# %%
# get standard configurations
cfg = get_cfg()

# select the right network
if mask_it:
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )

# no masking required?
else:
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"

# select datasets
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_valid",)
cfg.TEST.EVAL_PERIOD = 20
cfg.DATALOADER.NUM_WORKERS = 0

# training parameters
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

# roi and classes
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# set to "RepeatFactorTrainingSampler" in order to allow balanced sampling
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"

# below this threshold of occurency the rare samples are repeated
cfg.DATALOADER.REPEAT_THRESHOLD = 0.2

# larger image size profitable?
#cfg.INPUT.MIN_SIZE_TRAIN = (928,)

cfg.OUTPUT_DIR = "../output"

# Color and Class definitions

mal_col = (255, 69, 0)  # red
ben_col = (50, 205, 50)  # green


MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_colors = [
    ben_col,
    mal_col,
]

# %%

# Select Trainer
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
# %%
trainer.train()
# %%
