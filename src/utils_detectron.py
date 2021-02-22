# define some useful functionalities for detectron2
import numpy as np
import os
import torch
import copy
import math
import random
from PIL import Image
from torchvision.transforms import functional as F
# detectron core specific
from fvcore.common.file_io import PathManager
from fvcore.transforms.transform import Transform
# detectron specific
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator
import logging
from detectron2.data.transforms.augmentation import TransformGen


# personal functionalityfrom detectron2.evaluation import DatasetEvaluator


# use this dataloader instead of the default


class MyEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predicitions = []

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predicitions = []

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class CocoTrainer(DefaultTrainer):
    """
    customized training class, overwriteing some default functionalities
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """add the idividual train_loader:"""
        return get_dataloader(cfg, is_train=True)


class CocoTrainer2(DefaultTrainer):
    """
    customized training class, overwriteing some default functionalities
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """add the idividual train_loader:"""
        return get_dataloader(cfg, is_train=True)

# %% Apply the Rotation:


class RotTransform(Transform):
    """
    Perform rotation on image
    """

    def __init__(self, degree, h, w):
        super().__init__()
        self.degree, self.h, self.w = degree, h, w
        self.center_x = w // 2
        self.center_y = h // 2
        self.sind = math.sin(degree * (math.pi / 180))
        self.cosd = math.cos(degree * (math.pi / 180))

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Rotate the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # to PIL image
        img = Image.fromarray(img)

        # rotate the whole Image
        img = F.rotate(img, self.degree)
        # back to numpy:
        img = np.asarray(img)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Rotate the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        # x' = cos(alp) * x_c - sin(alp) * y_c
        x_new_c = self.cosd * (coords[:, 0] - self.center_x) + \
            self.sind * (coords[:, 1] - self.center_y)
        # y' = sin(alp) * x_c + cos(alp) * y_c
        y_new_c = - self.sind * (coords[:, 0] - self.center_x) + \
            self.cosd * (coords[:, 1] - self.center_y)

        # reapply to edge
        coords[:, 0] = x_new_c + self.center_x
        coords[:, 1] = y_new_c + self.center_y

        return coords


class RandomRot(TransformGen):
    """
    Randomly rotate the image and annotations
    """

    def __init__(self, deg_range=30):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        self.deg_range = deg_range

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        return random.uniform(degrees[0], degrees[1])

    def get_transform(self, img):
        # get the height / width
        h, w = img.shape[:2]
        # get random angle in range
        ang = self.get_params([-self.deg_range, self.deg_range])
        # return rotation operation
        return RotTransform(ang, h=h, w=w)


class MyDatasetMapper(object):
    """
    Customized Datasetmapper, strongly based on the default one:
    https://detectron2.readthedocs.io/_modules/detectron2/data/dataset_mapper.html#DatasetMapper
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(
                cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info(
                "CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict


class CocoTrainer2(DefaultTrainer):
    """
    customized training class, overwriteing some default functionalities
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(
            dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        """add the idividual train_loader:"""
        return get_dataloader(cfg, is_train=True)


def get_dataloader(cfg, is_train):
    """
    summarize all functionality in the dataloader
    """
    mapper = MyDatasetMapper(cfg, is_train)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    return data_loader


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    # Print the transformations
    logger = logging.getLogger(__name__)
    tfm_gens = []

    # always set to uniform scale
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    # add all the personalized transformations for training here:
    if is_train:
        # Crop
        tfm_gens.insert(0, (T.RandomCrop(
            crop_type="relative_range", crop_size=[0.6, 1])))
        # Horizontal
        tfm_gens.append(T.RandomFlip(horizontal=True))
        # Vertical
        tfm_gens.append(T.RandomFlip(horizontal=False, vertical=True))
        # Lightning
        tfm_gens.append(T.RandomLighting(scale=3))
        # Brightness
        tfm_gens.append(T.RandomBrightness(0.7, 1.3))
        # Contrast
        tfm_gens.append(T.RandomContrast(0.7, 1.3))
        # Intensity
        tfm_gens.append(T.RandomSaturation(
            intensity_min=0.6, intensity_max=1.4))
        # NEW: Rotation
        tfm_gens.append(RandomRot(deg_range=120))

        logger.info("TransformGens used in training: " + str(tfm_gens))

        print(tfm_gens)

    return tfm_gens
