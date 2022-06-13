import numpy as np


class LossComb:
    STANDARD = {"iou": ([1., 1., 1.], "CiouLoss"), "object": ([1., 1., 1.], "BoxObjectnessLoss", 1, 1),
                "category": ([1., 1., 1.], "MajorCategoryLoss")}


class TrainingPlan:
    KITTI_SIMPLE = [
        ("kitti", 10, 0.0001, LossComb.STANDARD, True),
        ("kitti", 50, 0.00001, LossComb.STANDARD, True)
    ]


class TfrParams:
    MIN_PIX = {'train': {"Pedestrian": 0, "Car": 0, "Cyclist": 0,
                         },
               'val': {"Pedestrian": 0, "Car": 0, "Cyclist": 0,
                       }
               }

    CATEGORY_NAMES = {"category": ["Pedestrian", "Car", "Cyclist"],
                      "dont": ["DontCare"]
                      }


class TrainParams:
    @classmethod
    def get_pred_composition(cls, iou_aware, categorized=False, depth=False):
        cls_composition = {"category": len(TfrParams.CATEGORY_NAMES["major"])}
        if depth:
            reg_composition = {"yxhwl": 5, "depth": 1, "theta": 1}
        else:
            reg_composition = {"yxhw": 4, "object": 1}
        if iou_aware:
            reg_composition["ioup"] = 1
        composition = {"reg": reg_composition, "cls": cls_composition}

        if categorized:
            out_composition = {name: sum(list(subdic.values())) for name, subdic in composition.items()}
        else:
            out_composition = dict()
            for names, subdic in composition.items():
                out_composition.update(subdic)

        return out_composition


assert list(TfrParams.MIN_PIX["train"].keys()) == TfrParams.CATEGORY_NAMES["category"]
assert list(TfrParams.MIN_PIX["val"].keys()) == TfrParams.CATEGORY_NAMES["category"]
