import numpy as np
import cv2

import dataloader.data_util as tu
import dataloader.preprocess as pr
import config as cfg


class ExampleMaker:
    def __init__(self, data_reader, dataset_cfg, split,
                 feat_scales=cfg.ModelOutput.FEATURE_SCALES,
                 category_names=cfg.Dataloader.CATEGORY_NAMES,
                 max_bbox=cfg.Dataloader.MAX_BBOX_PER_IMAGE,
                 max_dontcare=cfg.Dataloader.MAX_DONT_PER_IMAGE):
        self.data_reader = data_reader
        self.feat_scales = feat_scales
        self.category_names = category_names
        self.max_bbox = max_bbox
        self.max_depth = dataset_cfg.MAX_DEPTH
        self.preprocess_example = pr.ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                                       max_bbox=max_bbox,
                                                       max_dontcare=max_dontcare,
                                                       max_depth=dataset_cfg.MAX_DEPTH,
                                                       min_depth=dataset_cfg.MIN_DEPTH
                                                       )

    def get_example(self, index):
        example = dict()
        example["image"] = self.data_reader.get_image(index)
        example["depth"] = self.data_reader.get_depth(index)
        raw_hw_shape = example["image"].shape[:2]
        bboxes, categories = self.data_reader.get_bboxes(index, raw_hw_shape)
        example["bboxes"], example["dontcare"] = self.merge_box_and_category(bboxes, categories)
        example = self.preprocess_example(example)
        if index % 100 == 10:
            self.show_example(example)
        return example

    def merge_box_and_category(self, bboxes, categories):
        reamapped_categories = []
        for category_str in categories:
            if category_str in self.category_names["category"]:
                major_index = self.category_names["category"].index(category_str)
            elif category_str in self.category_names["dont"]:
                major_index = -1
            else:
                major_index = -2
            reamapped_categories.append(major_index)
        reamapped_categories = np.array(reamapped_categories)[..., np.newaxis]
        # bbox: yxhw, obj, ctgr (6)
        bboxes = np.concatenate([bboxes, reamapped_categories], axis=-1)
        dontcare = bboxes[bboxes[..., -1] == -1]
        bboxes = bboxes[bboxes[..., -1] >= 0]
        return bboxes, dontcare

    def show_example(self, example):
        image = tu.draw_boxes(example["image"], example["bboxes"], self.category_names)
        depth = example["depth"]
        depth[depth > self.max_depth] = self.max_depth
        depth = cv2.applyColorMap(depth.astype(np.uint8)*5, cv2.COLORMAP_TURBO)
        cv2.imshow("image with bboxes", image)
        cv2.imshow("depth", depth)
        cv2.waitKey(100)

