"""
学習用 custom dataset mapper。

設計方針（baseline_maskdino_design.md §2.4, §4.1）:
  - 元解像度(2048)から 1024x1024 をランダムクロップ（縮小しない）
  - 軽いスケールジッタ [0.7, 1.3]、90度単位の回転、H/V フリップ
  - brightness / contrast ジッタ（観測所間ドメインシフト対策）
  - 空クロップ（GT 0 個）はリトライ

MaskDINO の COCOInstanceNewBaselineDatasetMapper をベースにアノテーション処理を踏襲。
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, PolygonMasks
from pycocotools import mask as coco_mask

__all__ = ["FilamentInstanceDatasetMapper"]


def _polys_to_bitmask(polys, h, w):
    masks = []
    for poly in polys:
        rles = coco_mask.frPyObjects(poly, h, w)
        m = coco_mask.decode(rles)
        if m.ndim < 3:
            m = m[..., None]
        masks.append(torch.as_tensor(m, dtype=torch.uint8).any(dim=2))
    if masks:
        return torch.stack(masks, dim=0)
    return torch.zeros((0, h, w), dtype=torch.uint8)


def build_augmentation(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE          # 1024
    min_scale = cfg.INPUT.MIN_SCALE            # 0.7
    max_scale = cfg.INPUT.MAX_SCALE            # 1.3
    native = bool(getattr(cfg.INPUT, "NATIVE_CROP", False))
    augs = [
        T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=0.5),
        T.RandomApply(T.RandomContrast(0.8, 1.2), prob=0.5),
    ]
    if native:
        # 原寸クロップ。ResizeScale(target=image_size) は 2048 画像全体を image_size 付近へ
        # 縮小してしまう（実測: 面積中央 960px -> 188-386px、約2倍の縮小）。
        # 推論はフル2048の原寸で行うため、物体スケールが学習と推論で2倍ずれていた。
        # ここでは縮小せず、原寸のまま image_size の窓を切る。
        # スケール拡張は 2048 を基準に軽微な範囲だけ掛ける（1.0 付近）。
        if min_scale != 1.0 or max_scale != 1.0:
            augs.append(T.ResizeScale(min_scale=min_scale, max_scale=max_scale,
                                      target_height=2048, target_width=2048))
        augs.append(T.RandomCrop(crop_type="absolute", crop_size=(image_size, image_size)))
    else:
        augs.append(T.ResizeScale(min_scale=min_scale, max_scale=max_scale,
                                  target_height=image_size, target_width=image_size))
        augs.append(T.FixedSizeCrop(crop_size=(image_size, image_size)))
    augs += [
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomFlip(horizontal=False, vertical=True),
        T.RandomRotation(angle=[0.0, 90.0, 180.0, 270.0], expand=False, sample_style="choice"),
    ]
    return augs


class FilamentInstanceDatasetMapper:
    @configurable
    def __init__(self, is_train, *, augmentations, image_format, empty_retry=8):
        self.augs = T.AugmentationList(augmentations)
        self.is_train = is_train
        self.img_format = image_format
        self.empty_retry = empty_retry
        logging.getLogger(__name__).info(f"[FilamentMapper] augs: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        return {
            "is_train": is_train,
            "augmentations": build_augmentation(cfg),
            "image_format": cfg.INPUT.FORMAT,
        }

    def _apply_once(self, dataset_dict, image):
        aug_input = T.AugInput(image)
        transforms = self.augs(aug_input)
        out_img = aug_input.image
        shape = out_img.shape[:2]

        annos = [
            utils.transform_instance_annotations(obj, transforms, shape)
            for obj in dataset_dict.get("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]
        # 疑似ラベル（src/make_pseudo.py）は RLE マスクのまま持つ。ポリゴンに直すと
        # 細いフィラメントの輪郭画素が落ちて往復 IoU が 0.91 まで下がるため。
        # 本物の正解（ポリゴン）は従来どおり polygon 形式で処理する。
        fmt = "bitmask" if dataset_dict.get("pseudo") else "polygon"
        instances = utils.annotations_to_instances(annos, shape, mask_format=fmt)
        if not instances.has("gt_masks"):
            instances.gt_masks = PolygonMasks([])
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = utils.filter_empty_instances(instances)
        return out_img, instances

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        out_img, instances = self._apply_once(dataset_dict, image)
        tries = 0
        while self.is_train and len(instances) == 0 and tries < self.empty_retry:
            out_img, instances = self._apply_once(dataset_dict, image)
            tries += 1

        h, w = out_img.shape[:2]
        if instances.has("gt_masks") and isinstance(instances.gt_masks, PolygonMasks):
            instances.gt_masks = _polys_to_bitmask(instances.gt_masks.polygons, h, w)
        elif instances.has("gt_masks") and isinstance(instances.gt_masks, BitMasks):
            instances.gt_masks = instances.gt_masks.tensor.to(torch.uint8)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(out_img.transpose(2, 0, 1))
        )
        dataset_dict["instances"] = instances
        dataset_dict.pop("annotations", None)
        return dataset_dict
