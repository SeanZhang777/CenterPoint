import pathlib
import pickle
import re
import numpy as np

from collections import OrderedDict
from pathlib import Path
from skimage import io
from tqdm import tqdm

from det3d.core import box_np_ops


def convert_to_maxus_info_version2(info):
    """convert maxus info v1 to v2 if possible.
    """
    if "image" not in info or "calib" not in info or "point_cloud" not in info:
        info["image"] = {
            "image_shape": info["img_shape"],
            "image_idx": info["image_idx"],
            "image_path": info["img_path"],
        }
        info["calib"] = {
            "R0_rect": info["calib/R0_rect"],
            "Tr_velo_to_cam": info["calib/Tr_velo_to_cam"],
            "P2": info["calib/P2"],
        }
        info["point_cloud"] = {
            "velodyne_path": info["velodyne_path"],
        }


def maxus_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno["metadata"]["image_idx"]
        label_lines = []
        for j in range(anno["bbox"].shape[0]):
            label_dict = {
                "name": anno["name"][j],
                "alpha": anno["alpha"][j],
                "bbox": anno["bbox"][j],
                "location": anno["location"][j],
                "dimensions": anno["dimensions"][j],
                "rotation_y": anno["rotation_y"][j],
                "score": anno["score"][j],
            }
            label_line = maxus_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f"{get_image_index_str(image_idx)}.txt"
        label_str = "\n".join(label_lines)
        with open(label_file, "w") as f:
            f.write(label_str)


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def _calculate_num_points_in_gt(
    data_path, infos, relative_path, remove_outside=True, num_features=4
):
    for info in tqdm(infos):
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]
        if relative_path:
            v_path = str(Path(data_path) / pc_info["pointcloud_path"])
        else:
            v_path = pc_info["pointcloud_path"]
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(
            [-1, num_features]
        )
        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        gt_boxes_lidar = annos['box3d_lidar']
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['box3d_lidar']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def create_maxus_info_file(data_path, save_path=None, relative_path=True):
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / "train.txt"))
    val_img_ids = _read_imageset_file(str(imageset_folder / "val.txt"))

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    maxus_infos_train = get_maxus_image_info(
        data_path,
        training=True,
        pointcloud=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        with_imageshape=False
    )
    _calculate_num_points_in_gt(data_path, maxus_infos_train, relative_path)
    filename = save_path / "maxus_infos_train.pkl"
    print(f"Maxus info train file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(maxus_infos_train, f)
    maxus_infos_val = get_maxus_image_info(data_path,
                                           training=True,
                                           pointcloud=True,
                                           calib=True,
                                           image_ids=val_img_ids,
                                           relative_path=relative_path,
                                           with_imageshape=False)
    _calculate_num_points_in_gt(data_path, maxus_infos_val, relative_path)
    filename = save_path / 'maxus_infos_val.pkl'
    print(f"Maxus info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(maxus_infos_val, f)
    filename = save_path / 'maxus_infos_trainval.pkl'
    print(f"Maxus info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(maxus_infos_train + maxus_infos_val, f)

def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), all_pairs_min_ymax - all_pairs_max_ymin
    )

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), all_pairs_min_xmax - all_pairs_max_xmin
    )
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def get_image_index_str(img_idx):
    return img_idx


def get_maxus_info_path(
    idx,
    prefix,
    info_type="image",
    file_tail=".jpg",
    training=True,
    relative_path=True,
    exist_check=True,
):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = pathlib.Path("training") / info_type / img_idx_str
    else:
        file_path = pathlib.Path("testing") / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_maxus_info_path(
        idx, prefix, "image", ".png", training, relative_path, exist_check
    )


def get_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_maxus_info_path(
        idx, prefix, "label", ".txt", training, relative_path, exist_check
    )


def get_pointcloud_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_maxus_info_path(
        idx, prefix, "pointcloud", ".bin", training, relative_path, exist_check
    )


def get_calib_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_maxus_info_path(
        idx, prefix, "calib", ".txt", training, relative_path, exist_check
    )


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    return mat


def _check_maxus_directory(root_path):
    path = pathlib.Path(root_path)
    results = []
    results.append((path / "training").exists())
    results.append((path / "testing").exists())
    path_train_image_2 = path / "training" / "image_2"
    results.append(path_train_image_2.exists())
    results.append(len(path_train_image_2.glob("*.png")) == 7481)
    path_train_label_2 = path / "training" / "label_2"
    results.append(path_train_label_2.exists())
    path_train_lidar = path / "training" / "velodyne"
    results.append(path_train_lidar.exists())
    path_train_calib = path / "training" / "calib"
    results.append(path_train_calib.exists())
    results.append(len(path_train_label_2.glob("*.txt")) == 7481)
    results.append(len(path_train_lidar.glob("*.bin")) == 7481)
    results.append(len(path_train_calib.glob("*.txt")) == 7481)
    path_test_image_2 = path / "testing" / "image_2"
    results.append(path_test_image_2.exists())
    results.append(len(path_test_image_2.glob("*.png")) == 7518)
    path_test_lidar = path / "testing" / "velodyne"
    results.append(path_test_lidar.exists())
    path_test_calib = path / "testing" / "calib"
    results.append(path_test_calib.exists())
    results.append(len(path_test_lidar.glob("*.bin")) == 7518)
    results.append(len(path_test_calib.glob("*.txt")) == 7518)
    return np.array(results, dtype=np.bool)


def get_maxus_image_info(
    path,
    training=True,
    label_info=True,
    pointcloud=False,
    calib=False,
    image_ids=7481,
    extend_matrix=True,
    num_worker=8,
    relative_path=True,
    with_imageshape=True,
):
    # image_infos = []
    """
    MAXUS annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for maxus]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            pointcloud_path: ...
        }
        [optional, for maxus]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: maxus difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    import concurrent.futures as futures

    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {"num_features": 4}
        calib_info = {}

        image_info = {"image_idx": idx}
        annotations = None
        if pointcloud:
            pc_info["pointcloud_path"] = get_pointcloud_path(
                idx, path, training, relative_path
            )
        # image_info["image_path"] = get_image_path(idx, path, training, relative_path)
        if with_imageshape:
            img_path = image_info["image_path"]
            if relative_path:
                img_path = str(root_path / img_path)
            image_info["image_shape"] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32
            )
        else:
            image_info['image_path'] = None
            image_info['image_shape'] = np.array([1280, 720])
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info["image"] = image_info
        info["point_cloud"] = pc_info
        if calib:
            info['calib'] = calib_info

        if annotations is not None:
            info["annos"] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        infos = executor.map(map_func, image_ids)
    return list(infos)

def label_str_to_int(labels, remove_dontcare=True, dtype=np.int32):
    class_to_label = get_class_to_label_map()
    ret = np.array([class_to_label[l] for l in labels], dtype=dtype)
    if remove_dontcare:
        ret = ret[ret > 0]
    return ret


def get_class_to_label_map():
    class_to_label = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
        "Van": 3,
        "Person_sitting": 4,
        "Truck": 5,
        "Tram": 6,
        "Misc": 7,
        "DontCare": -1,
    }
    return class_to_label


def get_classes():
    return get_class_to_label_map().keys()


def filter_gt_boxes(gt_boxes, gt_labels, used_classes):
    mask = np.array([l in used_classes for l in gt_labels], dtype=np.bool)
    return mask


def filter_anno_by_mask(image_anno, mask):
    img_filtered_annotations = {}
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][mask]
    return img_filtered_annotations


def filter_infos_by_used_classes(infos, used_classes):
    new_infos = []
    for info in infos:
        annos = info["annos"]
        name_in_info = False
        for name in used_classes:
            if name in annos["name"]:
                name_in_info = True
                break
        if name_in_info:
            new_infos.append(info)
    return new_infos


def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno["name"]) if x != "DontCare"
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][relevant_annotation_indices]
    return img_filtered_annotations


def remove_low_height(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno["bbox"]) if (s[3] - s[1]) >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][relevant_annotation_indices]
    return img_filtered_annotations


def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, s in enumerate(image_anno["score"]) if s >= thresh
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][relevant_annotation_indices]
    return img_filtered_annotations


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def apply_mask_(array_dict):
    pass


def filter_kitti_anno(
    image_anno, used_classes, used_difficulty=None, dontcare_iou=None
):
    if not isinstance(used_classes, (list, tuple, np.ndarray)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno["name"]) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][relevant_annotation_indices]
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i
            for i, x in enumerate(img_filtered_annotations["difficulty"])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = img_filtered_annotations[key][
                relevant_annotation_indices
            ]

    if "DontCare" in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations["name"]) if x == "DontCare"
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations["bbox"]
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)
                ]
    return img_filtered_annotations


def filter_annos_class(image_annos, used_class):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno["name"]) if x in used_class
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = anno[key][relevant_annotation_indices]
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_score(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno["score"]) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = anno[key][relevant_annotation_indices]
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_difficulty(image_annos, used_difficulty):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(anno["difficulty"]) if x in used_difficulty
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = anno[key][relevant_annotation_indices]
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_annos_low_height(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno["bbox"]) if (s[3] - s[1]) >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = anno[key][relevant_annotation_indices]
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos


def filter_empty_annos(image_annos):
    new_image_annos = []
    for anno in image_annos:
        if anno["name"].shape[0] > 0:
            new_image_annos.append(anno.copy())

    return new_image_annos


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict(
        [
            ("name", None),
            ("truncated", -1),
            ("occluded", -1),
            ("alpha", -10),
            ("bbox", None),
            ("dimensions", [-1, -1, -1]),
            ("location", [-1000, -1000, -1000]),
            ("rotation_y", -10),
            ("score", 0.0),
        ]
    )
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == "name":
            res_line.append(val)
        elif key in ["truncated", "alpha", "rotation_y", "score"]:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == "occluded":
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append("{}".format(val))
        elif key in ["dimensions"]:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                val = [val[1], val[2], val[0]]
                res_line += [prec_float.format(v) for v in val]
        elif key in ["bbox", "location"]:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(res_dict.keys()))
    return " ".join(res_line)


def annos_to_kitti_label(annos):
    num_instance = len(annos["name"])
    result_lines = []
    for i in range(num_instance):
        result_dict = {
            "name": annos["name"][i],
            "truncated": annos["truncated"][i],
            "occluded": annos["occluded"][i],
            "alpha": annos["alpha"][i],
            "bbox": annos["bbox"][i],
            "dimensions": annos["dimensions"][i],
            "location": annos["location"][i],
            "rotation_y": annos["rotation_y"][i],
            "score": annos["score"][i],
        }
        line = kitti_result_line(result_dict)
        result_lines.append(line)
    return result_lines


def add_difficulty_to_annos(info):
    annos = info['annos']
    occlusion = annos['occluded']
    diff = occlusion
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'occluded': [],
        'bbox': [],
        'box3d_camera': [],
        'box3d_lidar': [],
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['occluded'] = np.array([int(x[1]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[2:6]]
                                    for x in content]).reshape(-1, 4)
    # normal lidar coordinate: x, y, z, l, w, h, yaw
    gt_box3d_lidar = np.array([[float(info) for info in x[6:13]]
                              for x in content]).reshape(-1, 7)
    annotations['box3d_camera'] = box_np_ops.pcdet_box3d_lidar2camera(
                                     gt_box3d_lidar)
    # very strange coordinate defination: x, y, z, w, l, h, rotation_y(cameara)
    gt_box3d_lidar = gt_box3d_lidar[:, [0, 1, 2, 4, 3, 5, 6]]
    # gt_box3d_lidar[:, 2] = gt_box3d_lidar[:, 2] - gt_box3d_lidar[:, 5] / 2.0  # bottom center
    gt_box3d_lidar[:, 6] = annotations['box3d_camera'][:, 6]
    annotations['box3d_lidar'] = gt_box3d_lidar

    if len(content) != 0 and len(content[0]) == 14:  # have score
        annotations['score'] = np.array([float(x[13]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_pseudo_label_anno():
    annotations = {}
    annotations.update(
        {
            "name": np.array(["Car"]),
            "truncated": np.array([0.0]),
            "occluded": np.array([0]),
            "alpha": np.array([0.0]),
            "bbox": np.array([[0.1, 0.1, 15.0, 15.0]]),
            "dimensions": np.array([[0.1, 0.1, 15.0, 15.0]]),
            "location": np.array([[0.1, 0.1, 15.0]]),
            "rotation_y": np.array([[0.1, 0.1, 15.0]]),
        }
    )
    return annotations


def get_start_result_anno():
    annotations = {}
    annotations.update(
        {
            # 'index': None,
            "name": [],
            "occluded": [],
            "bbox": [],
            "box3d_camera": [],
            "box3d_lidar": [],
            "score": [],
        }
    )
    return annotations


def empty_result_anno():
    annotations = {}
    annotations.update(
        {
            "name": np.array([]),
            "occluded": np.array([]),
            "bbox": np.zeros([0, 4]),
            "box3d_camera": np.zeros([0, 7]),
            "box3d_lidar": np.zeros([0, 7]),
            "score": np.array([]),
        }
    )
    return annotations


def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob("*.txt")
        prog = re.compile(r"^\d{6}.txt$")
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx_str = get_image_index_str(idx)
        label_filename = label_folder / (image_idx_str + ".txt")
        anno = get_label_anno(label_filename)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos


def anno_to_rbboxes(anno):
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes
