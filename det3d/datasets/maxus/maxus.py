import numpy as np
import pickle
import os

from copy import deepcopy

from det3d.core import box_np_ops
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS

from .maxus_common import *
from .eval_maxus import maxus_eval


@DATASETS.register_module
class MaxusDataset(PointCloudDataset):

    NumPointFeatures = 4

    def __init__(
        self,
        root_path,
        info_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        **kwargs
    ):
        super(MaxusDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode
        )
        assert self._info_path is not None
        if not hasattr(self, "_maxus_infos"):
            with open(self._info_path, "rb") as f:
                infos = pickle.load(f)
            self._maxus_infos = infos
        self._num_point_features = __class__.NumPointFeatures
        # print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names

    def __len__(self):
        if not hasattr(self, "_maxus_infos"):
            with open(self._info_path, "rb") as f:
                self._maxus_infos = pickle.load(f)

        return len(self._maxus_infos)

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._maxus_infos[0]:
            return None

        gt_annos = [info["annos"] for info in self._maxus_infos]

        return gt_annos

    def convert_detection_to_maxus_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [k for k in detection.keys()]
        gt_image_idxes = [str(info["image"]["image_idx"]) for info in self._maxus_infos]
        # print(f"det_image_idxes: {det_image_idxes[:10]}")
        # print(f"gt_image_idxes: {gt_image_idxes[:10]}")
        annos = []
        # for i in range(len(detection)):
        for det_idx in gt_image_idxes:
            det = detection[det_idx]
            info = self._maxus_infos[gt_image_idxes.index(det_idx)]
            # info = self._kitti_infos[i]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            anno = get_start_result_anno()
            num_example = 0

            if final_box_preds.shape[0] != 0:
                final_box_preds[:, -1] = box_np_ops.limit_period(
                    final_box_preds[:, -1], offset=0.5, period=np.pi * 2,
                )
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2

                # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
                # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera
                box3d_camera = box_np_ops.mmdet3d_box3d_lidar_to_camera(final_box_preds)
                camera_box_origin = [0.5, 1.0, 0.5]

                bbox = np.tile(np.array([0, 0, 1280, 720]), (box3d_camera.shape[0], 1))

                for j in range(box3d_camera.shape[0]):
                    image_shape = info["image"]["image_shape"]
                    if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                        continue
                    if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                        continue
                    bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                    bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                    anno["bbox"].append(bbox[j])

                    # anno["dimensions"].append(box3d_camera[j, [4, 5, 3]])
                    anno["box3d_camera"].append(box3d_camera[j])
                    anno["box3d_lidar"].append(final_box_preds[j])
                    anno["name"].append(class_names[int(label_preds[j])])
                    anno["occluded"].append(0)
                    anno["score"].append(scores[j])

                    num_example += 1

            if num_example != 0:
                anno = {n: np.array(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir=None, testset=None):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        """
        gt_annos = self.ground_truth_annotations
        dt_annos = self.convert_detection_to_maxus_annos(detections)

        metrics = maxus_eval(gt_annos, dt_annos, self._class_names)

        results = {
            "results": {
                "maxus": metrics["result"],
            },
            "detail": {
                "eval.maxus": {
                    "maxus": metrics["detail"],
                }
            },
        }

        return results, dt_annos

    def __getitem__(self, idx):
        return self.get_sensor_data(idx, with_gp=False)

    def get_sensor_data(self, idx, with_image=False, with_gp=False):

        info = self._maxus_infos[idx]

        if with_gp:
            gp = self.get_road_plane(idx)

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "ground_plane": -gp[-1] if with_gp else None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": MaxusDataset.NumPointFeatures,
                "image_idx": info["image"]["image_idx"],
                "image_shape": info["image"]["image_shape"],
                "token": str(info["image"]["image_idx"]),
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)

        # objgraph.show_growth(limit=3)
        # objgraph.get_leaking_objects()

        image_info = info["image"]
        image_path = image_info["image_path"]

        if with_image:
            image_path = self._root_path / image_path
            with open(str(image_path), "rb") as f:
                image_str = f.read()
            data["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": "png",
            }

        return data
