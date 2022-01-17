# %%

from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.utils.visualizer import ColorMode
import ipywidgets as widgets
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from get_labels import get_transformed_lines, draw_lines
from utils import plot_confusion_matrix
import torch
from sklearn.metrics import confusion_matrix

# import some common libraries
import numpy as np
import os
from detectron2.utils.logger import setup_logger
from utils_detectron import CocoTrainer2


def get_image_list(path):
    with open(path) as f:
        data = json.load(f)
    return [dat["file_name"] for dat in data["images"]]


def get_category_list(path):
    with open(path) as f:
        data = json.load(f)
    cat_list = []
    for cat in data["categories"]:
        cat_list.append(cat["name"])
    return cat_list


def filename_to_labelname(filename, path):
    """transform the filename to the labelname"""
    imagename = filename.split('/')[-1]
    labels = os.listdir(path)

    # slelect matching label
    for label in labels:
        if imagename in label:
            print(label.split('_')[1])
            return label

    return 'no matching label!'


def get_label_img(img, filename, path='../data/labels2'):
    """read label, and get the matching transformed lines"""
    # get the labelname
    label = filename_to_labelname(filename, path)
    complete_labelname = f'{path}/{label}'
    # extract lines from label
    lines = get_transformed_lines(img, complete_labelname)
    # draw lines on image
    pil_img = Image.fromarray(img)
    arr_img = draw_lines(pil_img, lines)
    return arr_img


def personal_score(mode='valid'):
    """evaluation of the detec model for class accuracy"""
    json_name = f'../{mode}.json'
    filenames = get_image_list(json_name)
    cat_list = get_category_list(json_name)

    # empty list of predictions and targets
    preds, targets = [], []
    count = 0
    for filename in filenames:
        # load image
        img = cv2.imread(filename)

        # predict
        with torch.no_grad():
            outputs = predictor(img)
            out = outputs["instances"].to("cpu")[:1]

        # get categorys
        pred_entity_int = out[:1].pred_classes[0]
        category_pred = cat_list[pred_entity_int]
        category_true = filename.split('/')[-2]

        count += 1 if category_pred == category_true else 0

        # append to
        preds.append(category_pred)
        targets.append(category_true)

    conf = confusion_matrix(targets, preds)
    acc = count / len(filenames)

    res = {
        'conf': conf,
        'acc': acc,
        'cat_list': cat_list
    }
    return res


def update(idx=10, bbox=True, mask=True, score=True, true_label=True, mode='test'):
    """Display the activations"""
    proposed = 1
    filename = get_image_list(f'../{mode}.json')[idx]

    img = cv2.imread(filename)
    outputs = predictor(img)
    v = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        instance_mode=ColorMode.SEGMENTATION,
    )

    instances = outputs["instances"].to("cpu")[:proposed]
    instances.remove("pred_masks") if not mask else None
    instances.pred_boxes = [[0, 0, 0, 0]] if not bbox else instances.pred_boxes
    instances.remove("scores") if not score else None

    v = v.draw_instance_predictions(instances)
    plt.figure(figsize=(8, 8))

    back_img = Image.fromarray(v.get_image()[:, :, ::-1])
    back_img = np.array(back_img)
    if true_label:
        back_img = get_label_img(back_img, filename)

    plt.imshow(back_img)
    return back_img


# %%
# General definitions:
mask_it = True
train = False

if train:
    setup_logger()

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
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"

# select datasets
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_valid",)
cfg.TEST.EVAL_PERIOD = 1000
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
cfg.DATALOADER.REPEAT_THRESHOLD = 0.3
cfg.OUTPUT_DIR = "../output3"

# Color and Class definitions
mal_col = (255, 69, 0)  # red
ben_col = (50, 205, 50)  # green


MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_colors = [
    ben_col,
    mal_col,
]

# Select Trainer
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer2(cfg)
trainer.resume_or_load(resume=False)
# %%
if train:
    trainer.train()
# %%


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
cfg.DATASETS.TEST = ("my_dataset_test",)
predictor = DefaultPredictor(cfg)

# %%
idx = widgets.IntSlider(8, 0, 30)
widgets.interactive(update, idx=idx)
# %%
res_valid = personal_score(mode='valid')
res_test = personal_score(mode='test')

fig1 = plot_confusion_matrix(res_valid['conf'], res_valid['cat_list'])
fig2 = plot_confusion_matrix(res_test['conf'], res_test['cat_list'])

# %%

evaluator = COCOEvaluator(
    "my_dataset_train", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "my_dataset_train")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# %%

evaluator = COCOEvaluator(
    "my_dataset_valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "my_dataset_valid")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# %%
evaluator = COCOEvaluator(
    "my_dataset_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# %%
