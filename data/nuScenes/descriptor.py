# coding=utf-8
from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import os
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import numpy as np
import open3d
import pickle
from collections import Counter
from num2words import num2words
from word2number import w2n
import copy
import math
import torch
import re
from tqdm import tqdm

ROOT_PATH = "/ihoment/youjie10/qwt/dataset/nuscenes/v1.0-trainval"
DESCRIPTION = "nuscenes_infos_10sweeps_description_test.pkl"
NUM = 34149
SEMANTIC_CLASS_NUM = 16

# ---- Tool Functions ----
def get_color():
    colors = [
        [0.8745, 0.4745, 0.4392],  # #DF7970 (223, 121, 112), 'car'
        [0.5569, 0.4784, 0.6392],  # #8E7AA3 (142, 122, 163), 'truck'
        [0.2980, 0.6118, 0.6275],  # #4C9CA0 ( 76, 156, 160), 'construction_vehicle'
        [0.3176, 0.8039, 0.6275],  # #51CDA0 ( 81, 205, 160), 'bus'
        [0.6824, 0.4902, 0.6   ],  # #AE7D99 (174, 125, 153), 'trailer'
        [0.9882, 0.7608, 0.4235],  # #FCC26C (252, 194, 108), 'barrier',
        [0.4275, 0.4706, 0.6784],  # #6D78AD (109, 120, 173), 'motorcycle'
        [0.3216, 0.7373, 0.6588],  # #52BCA8 ( 82, 188, 168), 'bicycle'
        [0.2980, 0.6118, 0.6275],  # #4C9CA0 ( 76, 156, 160), 'pedestrian'
        [0.8745, 0.4745, 0.4392],  # #DF7970 (223, 121, 112), 'traffic_cone'
        [0.7804, 0.4824, 0.5216],  # #C77B85 (199, 123, 133),
        [0.7882, 0.8745, 0.8745],  # #C9D45C (201, 212,  92)
        [0.8745, 0.5294, 0.3020],  # #DF874D (223, 135,  77)
        [0.3333, 0.5725, 0.6784],  # #5592AD ( 85, 146, 173)
        [0.7647, 0.5922, 0.3843],  # #C39762 (195, 151,  98)
        [0.6746, 0.8196, 0.4941],  # #8DD17E (141, 209, 126)
        [0.3176, 0.8039, 0.6275],  # #51CDA0 ( 81, 205, 160)
        [0.7098, 0.4745, 0.3216],  # #B57952 (181, 121,  82)
        [0.4275, 0.4706, 0.6784],  # #6D78AD (109, 120, 173)
        [0.8902, 0.7961, 0.3922],  # #E3CB64 (227, 203, 100)
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    return colors

def colorize_point_by_label(points, labels, name="0"):
    color_setting = get_color()

    colors = []
    for i in range(len(points)):
        if (labels[i] == 10):
            color = color_setting[-1]
        else:
            color = color_setting[labels[i]]

        # color = color_setting[labels[i]]
        colors.append(color)

    colors = np.asarray(colors)
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    pc.colors = open3d.utility.Vector3dVector(colors)

    name = f"pc_{name}.ply"
    open3d.io.write_point_cloud(filename=name, pointcloud=pc)

    print(f"Saving :{name}")

def class_name(index):
    names = {
        0:  "barrier",
        1:  "bicycle",
        2:  "bus",
        3:  "car",
        4:  "constructionvehicle",
        5:  "motorcycle",
        6:  "pedestrian",
        7:  "trafficcone",
        8:  "trailer",
        9:  "truck",
        10: "driveablesurface",
        11: "otherflat",
        12: "sidewalk",
        13: "terrain",
        14: "manmade",
        15: "vegetation",
        16: "ignore",
    }

    return names[index]

def get_learning_map(ignore_index=16):
    learning_map = {
        0: ignore_index,
        1: ignore_index,
        2: 6,
        3: 6,
        4: 6,
        5: ignore_index,
        6: 6,
        7: ignore_index,
        8: ignore_index,
        9: 0,
        10: ignore_index,
        11: ignore_index,
        12: 7,
        13: ignore_index,
        14: 1,
        15: 2,
        16: 2,
        17: 3,
        18: 4,
        19: ignore_index,
        20: ignore_index,
        21: 5,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        26: 12,
        27: 13,
        28: 14,
        29: ignore_index,
        30: 15,
        31: ignore_index,
    }
    return learning_map


def print_dict(infos):

    for k,v in infos.items():
        print(f"k : {k}, v : {infos[k]}")

def del_key(info, key):

    del info[key]
    return info

def most_frequent_element(lst):
    index_map = defaultdict(list)

    # 记录每个元素的索引
    for idx, val in enumerate(lst):
        index_map[val].append(idx)

    # 找到出现次数最多的元素
    most_common = max(index_map.items(), key=lambda x: len(x[1]))

    element = most_common[0]
    indices = most_common[1]
    count = len(indices)

    return element, count, indices

def search_same_name(all_descs):
    max_count = 0
    max_indices = []
    max_i = -1
    for i in range(len(all_descs)):
        objects = all_descs[i]
        _, count, indices = most_frequent_element(objects)
        if (count > max_count):
            max_count = count
            max_indices = indices
            max_i = i

    # 删除对应索引
    new_list = []
    for i in range(len(all_descs)):
        objects = all_descs[i]
        new_list = [x for i, x in enumerate(objects) if i not in max_indices]
        all_descs[i] = new_list

    max_count2 = 0
    max_indices2 = []
    max_i2 = -1
    if (len(new_list) > 0):
        for i in range(len(all_descs)):
            objects = all_descs[i]
            _, count, indices = most_frequent_element(objects)
            if (count > max_count2):
                max_count2 = count
                max_indices2 = indices
                max_i2 = i

    return max_i, max_indices, max_i2, max_indices2

def count_num(gt_names):
    counter = Counter(gt_names)
    return counter

def save_pkl(save_path=None, infos=None):
    if (save_path is None):
        save_path = f"{ROOT_PATH}/{DESCRIPTION}"
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
        print(f"---- Saving : {save_path} ----")

def read_pkl(
    file_path = f"{ROOT_PATH}/{DESCRIPTION}"
):
    with open(file_path, 'rb') as f:
        infos = pickle.load(f)

    return infos[:NUM]

def sort_dict_by_word_number(d):
    """
    将 {"one car":124, "nine cars":754, "three cars":111}
    按英文数字大小排序
    """
    return dict(
        sorted(
            d.items(),
            key=lambda item: w2n.word_to_num(item[0].split()[0])  # 提取 key 里的第一个英文数字
        )
    )

def count_cars(infos):
    cars = {}
    for i, info in enumerate(infos):
        car = info["description_class"][0]
        if (car not in cars.keys()):
            cars[car] = 1
        else:
            cars[car] = cars[car] + 1

    cars = sort_dict_by_word_number(cars)

    count = 0
    for car in cars.keys():
        count += cars[car]
        print(car, cars[car])

    print(f"count : {count}")

def count_keys():
    infos = read_pkl()

    count = {}
    for info in infos:
        text = info["text_l0"]
        if (text not in count.keys()):
            count[text] = 1
        else:
            count[text] += 1

    for key in count.keys():
        print(key, count[key])

def filter_text(text):
    if(text.__contains__(" no ")):
        return ""
    texts = re.split(r'(?<=[.!?])\s+', text.strip())[1:]
    temp_text = ""
    for i, text in enumerate(texts):
        if(i > 0):
            text = " "+text
        temp_text += text
    return temp_text

def check_point_in_box(pts, box):
    """
    pts[x,y,z]
    box[c_x,c_y,c_z,dx,dy,dz,heading]
"""

    shift_x = pts[0] - box[0]
    shift_y = pts[1] - box[1]
    shift_z = pts[2] - box[2]
    cos_a = math.cos(box[6])
    sin_a = math.sin(box[6])
    dx, dy, dz = box[3], box[4], box[5]
    local_x = shift_x * cos_a + shift_y * sin_a;
    local_y = shift_y * cos_a - shift_x * sin_a;
    if (abs(shift_z) > dz / 2.0 or abs(local_x) > dx / 2.0 or abs(local_y) > dy / 2.0):
        return False
    return True

def sort_number_word_phrases(phrases):
    """
    将类似 ["one car", "nine cars", "three cars"] 的列表按数字大小排序
    """
    return sorted(
        phrases,
        key=lambda p: w2n.word_to_num(p.split()[0])  # 提取第一个单词作为数字
    )

def del_keys(info, keys):
    for key in keys:
        del_key(info, key)

    return info

def get_text(info, key, func_name, dict, print_info=False):

    # Text Content
    info[key] = globals()[func_name](info)
    if(print_info): print(info[key])
    # Text Content

    # Sample Distribution
    value = info[key]
    if (not dict.keys().__contains__(value)):
        dict[value] = 1
    else:
        dict[value] += 1
    # Sample Distribution

    return info, dict

def get_point_cloud(lidar_path):
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
    return points
# ---- Tool Functions ----


# ---- Test Functions ----
def see_lidar_scene():
    dataroot = ROOT_PATH
    version = 'v1.0-trainval'

    # dataroot = "/data/qwt/dataset/nuscenes/raw_mini"
    # version = 'v1.0-mini'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    print(len(nusc.sample))  # 样本数量
    print(nusc.sample[0])  # 第一个样本的元数据

    # 获取第一个 sample
    sample = nusc.sample[0]

    scene_token = sample['scene_token']
    scene = nusc.get("scene", scene_token)

    firt_sample_token = scene["first_sample_token"]

    lidar_points_list = []
    i = 1
    while firt_sample_token != "":
        sample = nusc.get('sample', firt_sample_token)

        # 获取 LIDAR_TOP 数据
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_token)

        # 获取点云路径
        lidar_path = os.path.join(dataroot, lidar_sd['filename'])

        # 读取 .bin (x, y, z, intensity)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        lidar_points_list.append(points)

        # PC = open3d.geometry.PointCloud()
        # points = open3d.utility.Vector3dVector(points)
        # PC.points = points
        # open3d.io.write_point_cloud(f"points{i}.ply", PC)

        # 读取语义信息
        lidarseg_rec = nusc.get('lidarseg', lidar_token)
        lidarseg_rel_path = lidarseg_rec['filename']  # 相对路径
        lidarseg_path = os.path.join(dataroot, lidarseg_rel_path)
        learning_map = get_learning_map()
        segment = np.fromfile(
            str(lidarseg_path), dtype=np.uint8, count=-1
        ).reshape([-1])
        segment = np.vectorize(learning_map.__getitem__)(segment).astype(
            np.int64
        )

        print(segment.shape)

        # 下一个 sample
        sample_token = sample['next']
        i += 1

    return lidar_points_list

def get_description():
    dataroot = ROOT_PATH
    version = 'v1.0-trainval'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)


    infos = read_pkl()
    for i, info in enumerate(infos):
        print(f"----------  {i+1}/{len(infos)}  ----------")

        sample_token = info['token']
        sample = nusc.get("sample", sample_token)
        scene_token = sample["scene_token"]
        scene = nusc.get("scene", scene_token)
        description = scene["description"]

        info["description"] = description

        infos[i] = info

    save_pkl(infos=infos)

def check_text_num(key):
    num_dict = {}

    infos = read_pkl()[:-1]
    num = len(infos)
    for i, info in enumerate(tqdm(infos, total=num, desc="checking Text : ")):
        text = info[key]

        if(num_dict.keys().__contains__(text)):
            num_dict[text] += 1
        else:
            num_dict[text] = 1

    print(f"Test Key : {key}")
    for key in num_dict:
        print(f"({num_dict[key]}) {key}")

def get_num_everyclass():

    '''
        car : 33266
        pedestrian : 27862
        trafficcone : 14853
        truck : 24118
        driveablesurface : 34149
        otherflat : 15462
        sidewalk : 33549
        terrain : 29291
        manmade : 34149
        vegetation : 33425
        ignore : 34149
        constructionvehicle : 9495
        barrier : 12320
        motorcycle : 7518
        bicycle : 7474
        bus : 10986
        trailer : 9432

    '''

    class_num = {}
    infos = read_pkl()
    for info in infos:
        semantic = info["semantic"]

        for i in range(SEMANTIC_CLASS_NUM+1):
            if(semantic.__contains__(i)):
                name = class_name(i)
                if(class_num.keys().__contains__(name)):
                    class_num[name] += 1
                else:
                    class_num[name] = 1

    for key in class_num.keys():
        print(f"{key} : {class_num[key]}")
# ---- Test Functions ----

# ---- Generation Tool Function ----
def orientation_text_rad(boxes, two_ori=False):
    """
    将车辆 yaw（弧度制）转换为前/后/左/右朝向描述
    """
    yaw_rad = boxes[6]
    yaw_deg = math.degrees(yaw_rad) % 360  # 转为 0~360 度

    if(two_ori):
        if 315 <= yaw_deg or yaw_deg < 135:
            return "facing forward"
        else:
            return "facing backward"
    else:
        if 315 <= yaw_deg or yaw_deg < 45:
            return "facing forward"
        elif 45 <= yaw_deg < 135:
            return "facing left"
        elif 135 <= yaw_deg < 225:
            return "facing backward"
        else:
            return "facing right"

def relative_position(boxA, boxB, threshold=0.0):
    '''
        获取简单的boxA在boxB的所处位置
        boxA : 源盒子， (N,)
        boxB : 目标盒子，源盒子的参考系，(N,)
    '''

    dx = boxA[0] - boxB[0]
    dy = boxA[1] - boxB[1]

    # 前后
    if dx > threshold:
        pos_x = "ahead"
    elif dx < -threshold:
        pos_x = "behind"
    else:
        pos_x = "aligned"

    # 左右
    if dy > threshold:
        pos_y = "left"
    elif dy < -threshold:
        pos_y = "right"
    else:
        pos_y = "center"

    return pos_x, pos_y

def get_boxes_to_boxes(item, object="car"):
    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    # ---- 将目标盒子和其他类别盒子分离开 ----
    object_boxes = []
    other_boxes = []
    other_names = []
    for i in range(len(gt_names)):
        if (gt_names[i] == object):
            object_boxes.append(gt_boxes[i])
        else:
            other_boxes.append(gt_boxes[i])
            other_names.append(gt_names[i])
    # ---- 将目标盒子和其他类别盒子分离开 ----

    # ---- all_descs的首顺序是以：其他类别开始的，每个盒子，盒子位置 -----
    all_descs = []
    for i in range(len(other_boxes)):
        descs = []
        for j in range(len(object_boxes)):
            desc = relative_position(object_boxes[j], other_boxes[i])
            descs.append(desc)
        all_descs.append(descs)
    # ---- all_descs的首顺序是以：其他类别开始的，每个盒子，盒子位置 -----

    return all_descs, other_boxes, other_names, object_boxes

def get_class_label(gt_names, boxes):
    CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    class_labels = []
    boxes_index = []
    for i, gt_name in enumerate(gt_names):
        if (gt_name in CLASS_NAMES):
            label = CLASS_NAMES.index(gt_name)
            class_labels.append(label)
            boxes_index.append(i)

    boxes = boxes[boxes_index, :]
    class_label = np.expand_dims(np.asarray(class_labels), axis=1)
    boxes = np.concatenate([boxes, class_label], axis=-1)
    return boxes
# ---- Generation Tool Function ----

# ---- 合并nuscenes_infos_10sweeps_train.pkl和nuscenes_infos_10sweeps_val.pkl ----
def class_descripter(items):
    # 类别表
    CLASS_NAMES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]

    # 数字到英文的映射
    num_to_word = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
    }

    # 统计数量
    counter = Counter(items)

    # 按类别顺序生成描述
    descriptions = []

    for cls in CLASS_NAMES:
        count = counter.get(cls, 0)
        word_count = num2words(count)

        name_form = cls
        if (count > 1):
            if (cls == "bus"):
                # 简单复数处理
                name_form = cls + "es"
            else:
                # 简单复数处理
                name_form = cls + "s"
        descriptions.append(f"{word_count} {name_form}")

    downsample_description = "Downsampling Lidar with "
    upsampling_description = "Upsampling Lidar with "

    descriptions.append(downsample_description)
    descriptions.append(upsampling_description)

    return descriptions

def descripter(items):
    # 统计每个词出现的次数
    counter = Counter(items)

    # 数字到英文的映射
    num_to_word = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
    }

    # 生成描述
    descriptions = []
    for item, count in counter.items():

        word_form = item
        if (count > 1):
            if (item == "bus"):
                # 简单复数处理
                word_form = item + "es"
            else:
                # 简单复数处理
                word_form = item + "s"

        descriptions.append(f"{num2words(count)} {word_form}")

    # 拼接成最终句子
    result = ", ".join(descriptions) + "."

    return result

def generate_description(infos, split="train"):
    '''
        生成当前LiDAR场景中出现类型的数量文本描述
        Example:
            Two cars, three trucks, one tree, ... , zero barrier.
    '''

    num = len(infos)

    for i, info in enumerate(tqdm(infos, total=num, desc=f"Reading nuscenes_infos_10sweeps_{split}.pkl : ")):
        # print(f"----- {i}/{num} -----")
        gt_names = info["gt_names"]
        description = descripter(gt_names)
        infos[i]["description"] = description
        description_class = class_descripter(gt_names)
        infos[i]["description_class"] = description_class

    return infos

def read_nuscenes_infos_10sweeps(
        split="train"
):
    info_path = f"{ROOT_PATH}/nuscenes_infos_10sweeps_{split}.pkl"

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
        return infos


def read_nuscenes_dbinfos_10sweeps_withvelo():
    info_path = f"{ROOT_PATH}/nuscenes_dbinfos_10sweeps_withvelo.pkl"

    nuscenes_infos = []
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
        nuscenes_infos.extend(infos)

def generate_infos_train_val_pkl():
    '''
        合并以下内容:
            nuscenes_infos_10sweeps_train.pkl
            nuscenes_infos_10sweeps_val.pkl
            -> nuscenes_infos_10sweeps_description.pkl
    '''
    train_infos = generate_description(read_nuscenes_infos_10sweeps("train"), split="train")
    val_infos = generate_description(read_nuscenes_infos_10sweeps("val"), split="train")
    infos = train_infos + val_infos

    save_path = f"{ROOT_PATH}/{DESCRIPTION}"
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
        print(f"---- Saving : {save_path} ----")
# ---- 合并nuscenes_infos_10sweeps_train.pkl和nuscenes_infos_10sweeps_val.pkl ----



# ---- 获取原始nuScenes的Semantci和Description ----
def get_semantic_and_description():
    '''
        从nuScenes中读取:
            Semantic Label
            Description Label
    '''

    dataroot = ROOT_PATH
    version = 'v1.0-trainval'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    infos = read_pkl()
    num = len(infos)
    for i, info in enumerate(tqdm(infos, total=NUM, desc="Getting Initial Semantic and Description : ")):
        # print(f"----------  {i+1}/{len(infos)}  ----------")

        # ---- Getting Sample Token ----
        sample_token = info['token']
        sample = nusc.get("sample", sample_token)
        # ---- Getting Sample Token ----

        # ---- Getting Semantic Label ----
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidarseg_rec = nusc.get('lidarseg', lidar_token)
        lidarseg_rel_path = lidarseg_rec['filename']  # 相对路径
        lidarseg_path = os.path.join(dataroot, lidarseg_rel_path)
        learning_map = get_learning_map()
        segment = np.fromfile(
            str(lidarseg_path), dtype=np.uint8, count=-1
        ).reshape([-1])
        segment = np.vectorize(learning_map.__getitem__)(segment).astype(
            np.int64
        )
        info["semantic"] = segment
        # ---- Getting Semantic Label ----

        # ---- Getting Description Label ----
        scene_token = sample["scene_token"]
        scene = nusc.get("scene", scene_token)
        description = scene["description"]
        info["description"] = description
        # ---- Getting Description Label ----

        infos[i] = info

    save_pkl(infos=infos)
# ---- 获取原始nuScenes的Semantci和Description ----

# ---- 生成每个LiDAR场景的Text ----
def text_quantity_l1(item, object="car", threshold=2):
    '''
        Key: text_quantity_l1
        Using The Training and Combination.
        Exmaple : NUM_THRESHOLD = 2
            No cars.
            One car.
            Two cars.
            More than two cars.
    '''

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num == 0):
        text = f"No {object}{suffix}."
    elif (car_num == 1):
        text = f"One {object}."
    elif (car_num <= threshold):
        text = f"{car_num_word} {object}{suffix}."
        text = text.capitalize()
    else:
        text = f"More than {num2words(threshold)} {object}{suffix}."
    return text

def text_quantity_l2(item, object="car", threshold=2):
    '''
        Key: text_quantity_l2
        Using The Training and Combination.
        Exmaple : NUM_THRESHOLD = 2
            There are no cars in the scene.
            There is one car in the scene.
            There are two cars in the scene.
            There are more than two cars in the scene.
    '''

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num == 0):
        text = f"There are no {object}{suffix} in the scene."
    elif (car_num == 1):
        text = f"There is one {object} in the scene."
    elif (car_num <= threshold):
        text = f"There are {car_num_word} {object}{suffix} in the scene."
    else:
        text = f"There are more than {num2words(threshold)} {object}{suffix} in the scene."
    return text

def text_quantity_star(item, object="car", threshold=5):
    '''
        Key: text_quantity_star
        Using The Training and Combination.
        Exmaple : NUM_THRESHOLD = 5
            Less than five cars.
            More than five cars.
    '''

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(threshold)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num <= threshold):
        text = f"Less than {num2words(threshold)} {object}{suffix}."
    else:
        text = f"More than {num2words(threshold)} {object}{suffix}."
    return text

def text_location_l1(item, object="car"):
    '''
        Key: text_location_l1
        Using The Training.
        Exmaple :
            No cars.
            There are cars.
            One car is ahead to the right of one pedestrian.
            One car is behind to the right of one pedestrian.
            One car is ahead to the left of one pedestrian.
            One car is behind to the left of one pedestrian.

            CLASS_NAMES = ['car', 'truck', 'construction_vehicle',
            'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
            'pedestrian', 'traffic_cone']
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f"No {object}{suffix}."

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = f"There are {object}{suffix}."

    if (len(all_descs) > 0):
        text = f"One {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}."

    return text

def text_location_l1c(item, object="car"):
    '''
        Key: text_location_l1c
        Using The Content Combination.
        Exmaple :

            One car is ahead to the right of one pedestrian.
            One car is behind to the right of one pedestrian.
            One car is ahead to the left of one pedestrian.
            One car is behind to the left of one pedestrian.

            CLASS_NAMES = ['car', 'truck', 'construction_vehicle',
            'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
            'pedestrian', 'traffic_cone']
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}."

    return text

def text_location_star(
        item,
        object="car",
        keys = ["pedestrian", "barrier", "truck"]
):
    '''
        Key: text_location_star
        Using The Training.
        Exmaple :
            No cars.
            There are cars.
            One car is ahead one pedestrian.
            One car is ahead one barrier.
            One car is ahead one truck.
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f"No {object}{suffix}."

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = f"There are {object}{suffix}."

    if (len(all_descs) > 0):
        if(keys.__contains__(other_names[0])):
            text = f"One {object} is around one {other_names[0]}."

    return text

def text_location_starc(
        item,
        object="car",
        keys=["pedestrian", "barrier", "truck"]
):
    '''
        Key: text_location_starc
        Using The Content Combination.
        Exmaple :

            One car is ahead one pedestrian.
            One car is ahead one barrier.
            One car is ahead one truck.
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = ""

    if (len(all_descs) > 0):
        if (keys.__contains__(other_names[0])):
            text = f"One {object} is around one {other_names[0]}."

    return text

def text_orientation_l1(item, object="car"):
    '''
        Key: text_orientation_l1
        Using The Training and Combination.
        Exmaple :
            No cars.
            There are cars.
            One car is facing forward.
            One car is facing right.
            One car is facing backward.
            One car is facing left.
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f"No {object}{suffix}."

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = f"There are {object}{suffix}."

    if (len(all_descs) > 0):
        text = f"One {object} is {orientation_text_rad(object_boxes[0])}."


    return text

def text_orientation_l1c(item, object="car"):
    '''
        Key: text_orientation_l1
        Using The Training and Combination.
        Exmaple :

            One car is facing forward.
            One car is facing right.
            One car is facing backward.
            One car is facing left.
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is {orientation_text_rad(object_boxes[0])}."


    return text

def text_orientation_star(item, object="car"):
    '''
        Key: text_orientation_star
        Using The Training and Combination.
        Exmaple :
            No cars.
            There are cars.
            One car is facing right.
            One car is facing left.
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f"No {object}{suffix}."

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = f"There are {object}{suffix}."

    if (len(all_descs) > 0):
        text = f"One {object} is {orientation_text_rad(object_boxes[0], two_ori=True)}."

    return text

def text_orientation_starc(item, object="car"):
    '''
        Key: text_orientation_star
        Using The Training and Combination.
        Exmaple :

            One car is facing right.
            One car is facing left.
    '''

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f"No {object}{suffix}."

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    text = f"There are {object}{suffix}."

    if (len(all_descs) > 0):
        text = f"One {object} is {orientation_text_rad(object_boxes[0], two_ori=True)}."

    return text

def text_weather(info):
    '''
        Key: text_weather
        Using The Training and Combination.
        Exmaple :
            Rainy.
            Sunny.
    '''

    text = info["description"]
    if(text.__contains__("rain") or text.__contains__("Rain")):
        return "Rainy."
    else:
        return "Sunny."

def text_time(info):
    '''
        Key: text_time
        Using The Training and Combination.
        Exmaple :
            Night.
            Day.
    '''

    text = info["description"]
    if(text.__contains__("night") or text.__contains__("Night")):
        return "Night."
    else:
        return "Day."

def get_text_decription(print_info=False):
    '''
        生成不同类型的Text Description.
    '''

    text_quantity_l1_dict = {}
    text_quantity_l2_dict = {}
    text_quantity_star_dict = {}
    text_location_l1_dict = {}
    text_location_l1c_dict = {}
    text_location_star_dict = {}
    text_location_starc_dict = {}
    text_orientation_l1_dict = {}
    text_orientation_l1c_dict = {}
    text_orientation_star_dict = {}
    text_orientation_starc_dict = {}
    text_weather_dict = {}
    text_time_dict = {}
    text_aim_dict = {}

    num_info = {
        "text_quantity_l1" : text_quantity_l1_dict,
        "text_quantity_l2" : text_quantity_l2_dict,
        "text_quantity_star" : text_quantity_star_dict,
        "text_location_l1" : text_location_l1_dict,
        "text_location_l1c" : text_location_l1c_dict,
        "text_location_star" : text_location_star_dict,
        "text_location_starc" : text_location_starc_dict,
        "text_orientation_l1" : text_orientation_l1_dict,
        "text_orientation_l1c" : text_orientation_l1c_dict,
        "text_orientation_star" : text_orientation_star_dict,
        "text_orientation_starc" : text_orientation_starc_dict,
        "text_weather" : text_weather_dict,
        "text_time" : text_time_dict,
        "text_aim" : text_aim_dict,
    }

    infos = read_pkl()
    for i, info in enumerate(tqdm(infos, total=NUM, desc="Getting Text Decription : ")):
        # print(f"---- {i}/{len(infos)} ----")

        # info = del_key(info, "text_quantity_l1")
        # info = del_key(info, "text_quantity_l2")
        # info = del_key(info, "text_quantity_star")
        # info = del_key(info, "text_location_l1")
        # info = del_key(info, "text_location_l1c")
        # info = del_key(info, "text_location_star")
        # info = del_key(info, "text_location_starc")
        # info = del_key(info, "text_orientation_l1")
        # info = del_key(info, "text_orientation_l1c")
        # info = del_key(info, "text_orientation_star")
        # info = del_key(info, "text_orientation_starc")
        # info = del_key(info, "text_weather")
        # info = del_key(info, "text_time")

        # ---- text_quantity_l1 ----
        '''       
            Key: text_quantity_l1
            Using The Training and Combination.
            Exmaple : NUM_THRESHOLD = 2
                (857) No cars.
                (2099) Two cars.
                (1873) One car.        
                (29320) More than two cars.
        '''
        get_text(info, "text_quantity_l1", "text_quantity_l1", text_quantity_l1_dict, print_info)
        # ---- text_quantity_l1 ----

        # ---- text_quantity_l2 ----
        '''
            Key: text_quantity_l2
            Using The Training and Combination.
            Exmaple : NUM_THRESHOLD = 2
                (2099) There are two cars in the scene.
                (1873) There is one car in the scene.
                (857) There are no cars in the scene.
                (29320) There are more than two cars in the scene.   
        '''
        get_text(info, "text_quantity_l2", "text_quantity_l2", text_quantity_l2_dict)
        # ---- text_quantity_l2 ----

        # ---- text_quantity_star ----
        '''
            Key: text_quantity_star
            Using The Training and Combination.
            Exmaple : NUM_THRESHOLD = 5
                (10692) Less than five cars.
                (23457) More than five cars.       
        '''
        get_text(info, "text_quantity_star", "text_quantity_star", text_quantity_star_dict, print_info)
        # ---- text_quantity_star ----

        # ---- text_location_l1 ----
        '''
            Key: text_location_l1
            Using The Training.
            Exmaple :
                (857) No cars.
                (509) There are cars.
                (681) One car is ahead to the right of one traffic_cone.
                (478) One car is ahead to the right of one ignore.
                (3055) One car is behind to the right of one pedestrian.
                (3111) One car is ahead to the left of one pedestrian.
                (3077) One car is ahead to the right of one pedestrian.
                (924) One car is behind to the right of one barrier.
                (976) One car is ahead to the left of one barrier.
                (1501) One car is ahead to the left of one truck.
                (1458) One car is ahead to the right of one truck.
                (1772) One car is behind to the right of one truck.
                (2291) One car is behind to the left of one pedestrian.
                (691) One car is behind to the right of one traffic_cone.
                (1084) One car is behind to the left of one barrier.
                (357) One car is ahead to the right of one motorcycle.
                (264) One car is behind to the right of one motorcycle.
                (360) One car is ahead to the left of one motorcycle.
                (438) One car is behind to the right of one trailer.
                (1792) One car is behind to the left of one truck.
                (137) One car is behind to the left of one bicycle.
                (241) One car is behind to the left of one construction_vehicle.
                (835) One car is ahead to the right of one barrier.
                (148) One car is ahead to the left of one construction_vehicle.
                (735) One car is behind to the left of one traffic_cone.
                (300) One car is behind to the left of one motorcycle.
                (292) One car is ahead to the right of one bus.
                (401) One car is behind to the right of one bus.
                (162) One car is behind to the right of one bicycle.
                (471) One car is behind to the left of one ignore.
                (526) One car is ahead to the left of one ignore.
                (645) One car is ahead to the left of one traffic_cone.
                (296) One car is ahead to the right of one bicycle.
                (591) One car is behind to the right of one ignore.
                (203) One car is ahead to the left of one bicycle.
                (118) One car is ahead to the right of one construction_vehicle.
                (219) One car is behind to the right of one construction_vehicle.
                (550) One car is behind to the left of one bus.
                (378) One car is ahead to the right of one trailer.
                (418) One car is ahead to the left of one bus.
                (339) One car is ahead to the left of one trailer.
                (468) One car is behind to the left of one trailer.
    
                CLASS_NAMES = ['car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone']
        '''
        get_text(info, "text_location_l1", "text_location_l1", text_location_l1_dict, print_info)
        # ---- text_location_l1 ----

        # ---- text_location_l1c ----
        '''
            Key: text_location_l1c
            Using The Content Combination.
            Exmaple :
    
                One car is ahead to the right of one pedestrian.
                One car is behind to the right of one pedestrian.
                One car is ahead to the left of one pedestrian.
                One car is behind to the left of one pedestrian.
                
                (1366) 
                (681) One car is ahead to the right of one traffic_cone.
                (478) One car is ahead to the right of one ignore.
                (3055) One car is behind to the right of one pedestrian.
                (3111) One car is ahead to the left of one pedestrian.
                (3077) One car is ahead to the right of one pedestrian.
                (924) One car is behind to the right of one barrier.
                (976) One car is ahead to the left of one barrier.
                (1501) One car is ahead to the left of one truck.
                (1458) One car is ahead to the right of one truck.
                (1772) One car is behind to the right of one truck.
                (2291) One car is behind to the left of one pedestrian.
                (691) One car is behind to the right of one traffic_cone.
                (1084) One car is behind to the left of one barrier.
                (357) One car is ahead to the right of one motorcycle.
                (264) One car is behind to the right of one motorcycle.
                (360) One car is ahead to the left of one motorcycle.
                (438) One car is behind to the right of one trailer.
                (1792) One car is behind to the left of one truck.
                (137) One car is behind to the left of one bicycle.
                (241) One car is behind to the left of one construction_vehicle.
                (835) One car is ahead to the right of one barrier.
                (148) One car is ahead to the left of one construction_vehicle.
                (735) One car is behind to the left of one traffic_cone.
                (300) One car is behind to the left of one motorcycle.
                (292) One car is ahead to the right of one bus.
                (401) One car is behind to the right of one bus.
                (162) One car is behind to the right of one bicycle.
                (471) One car is behind to the left of one ignore.
                (526) One car is ahead to the left of one ignore.
                (645) One car is ahead to the left of one traffic_cone.
                (296) One car is ahead to the right of one bicycle.
                (591) One car is behind to the right of one ignore.
                (203) One car is ahead to the left of one bicycle.
                (118) One car is ahead to the right of one construction_vehicle.
                (219) One car is behind to the right of one construction_vehicle.
                (550) One car is behind to the left of one bus.
                (378) One car is ahead to the right of one trailer.
                (418) One car is ahead to the left of one bus.
                (339) One car is ahead to the left of one trailer.
                (468) One car is behind to the left of one trailer.
    
                CLASS_NAMES = ['car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone']        
        '''
        get_text(info, "text_location_l1c", "text_location_l1c", text_location_l1c_dict, print_info)
        # ---- text_location_l1c ----

        # ---- text_location_star ----
        '''
            Key: text_location_star
            Using The Training.
            Exmaple :
                (857) No cars.
                (11416) There are cars.
                (11534) One car is around one pedestrian.
                (3819) One car is around one barrier.
                (6523) One car is around one truck.
        '''
        get_text(info, "text_location_star", "text_location_star", text_location_star_dict, print_info)
        # ---- text_location_star ----

        # ---- text_location_starc ----
        '''
            Key: text_location_starc
            Using The Content Combination.
            Exmaple :    
                (12273) 
                (11534) One car is around one pedestrian.
                (3819)  One car is around one barrier.
                (6523)  One car is around one truck.   
        '''
        get_text(info, "text_location_starc", "text_location_starc", text_location_starc_dict, print_info)
        # ---- text_location_starc ----

        # ---- text_orientation_l1 ----
        '''
            Key: text_orientation_l1
            Using The Training.
            Exmaple :
                (857) No cars.
                (509) There are cars.
                (7416) One car is facing backward.
                (6662) One car is facing forward.
                (9994) One car is facing left.
                (8711) One car is facing right.
        '''
        get_text(info, "text_orientation_l1", "text_orientation_l1", text_orientation_l1_dict, print_info)
        # ---- text_orientation_l1 ----

        # ---- text_orientation_l1c ----
        '''
            Key: text_orientation_l1c
            Using The Content Combination.
            Exmaple :
                (1366) 
                (7416) One car is facing backward.
                (6662) One car is facing forward.
                (9994) One car is facing left.
                (8711) One car is facing right.
        '''
        get_text(info, "text_orientation_l1c", "text_orientation_l1c", text_orientation_l1c_dict, print_info)
        # ---- text_orientation_l1c ----

        # ---- text_orientation_star ----
        '''
            Key: text_orientation_star
            Using The Training.
            Exmaple :
                (857) No cars.
                (509) There are cars.
                (16127) One car is facing backward.
                (16656) One car is facing forward.         
        '''
        get_text(info, "text_orientation_star", "text_orientation_star", text_orientation_star_dict, print_info)
        # ---- text_orientation_star ----

        # ---- text_orientation_starc ----
        '''
            Key: text_orientation_starc
            Using The Content Combination.
            Exmaple :
                (1366) 
                (16127) One car is facing backward.
                (16656) One car is facing forward.         
        '''
        get_text(info, "text_orientation_starc", "text_orientation_starc", text_orientation_starc_dict, print_info)
        # ---- text_orientation_starc ----

        # ---- text_weather ----
        '''
            Key: text_weather
            Using The Training and Combination.
            Exmaple :
                (6670)  Rainy. 
                (27479) Sunny.      
        '''
        get_text(info, "text_weather", "text_weather", text_weather_dict, print_info)
        # ---- text_weather ----

        # ---- text_time ----
        '''
            Key: text_time
            Using The Training and Combination.
            Exmaple :
                (30162) Day.
                (3987)  Night.       
        '''
        get_text(info, "text_time", "text_time", text_time_dict, print_info)
        # ---- text_time ----

        # ---- text_aim ----
        '''
            Key: text_aim
            Using The Training.
            Example:
                (10101) Sunny.
                (4536) Sunny. One car is around one truck.
                (2966) Sunny. One car is around one barrier.
                (9876) Sunny. One car is around one pedestrian.
                (2172) Rainy.
                (1987) Rainy. One car is around one truck.
                (853) Rainy. One car is around one barrier.
                (1658) Rainy. One car is around one pedestrian.
        '''
        text_location_starc_value = info["text_location_starc"]
        if(text_location_starc_value == ""):
            info["text_aim"] = info["text_weather"]
        else:
            info["text_aim"] = info["text_weather"] + " " + info["text_location_starc"]

        text_aim_num = info["text_aim"]

        if(not text_aim_dict.keys().__contains__(text_aim_num)):
            text_aim_dict[text_aim_num] = 1
        else:
            text_aim_dict[text_aim_num] += 1
        info["text_aim_num"] = text_aim_dict
        # ---- text_aim----

        infos[i] = info

    infos.append(num_info)
    save_pkl(infos=infos)


def print_info(save_infos=False):
    infos = read_pkl()
    num_info = infos[NUM:]

    for key_i in num_info.keys():
        num_text = len(num_info[key_i])
        print(f"num : {num_text}")
        for key_j in num_info[key_i].keys():
            print(f"{key_j} : {num_info[key_i][key_j]}")
        if(save_infos): num_info[key_i]["num"] = len(num_info[key_i])
        print("=======================================")

    if(save_infos):
        infos[-1] = num_info
        save_pkl(infos=infos)

# ---- 生成每个LiDAR场景的Text ----

# ---- 一键生成 ----
def generate_descirption_pkl():
    '''
        已经存在以下两个文件:
            1. nuscenes_infos_10sweeps_train.pkl
            2. nuscenes_infos_10sweeps_val.pkl
        这将生成:
            nuscenes_infos_10sweeps_description.pkl

        Your Path:
            |--lidarseg
            |--maps
            |--samples
            |--sweeps
            |--v1.0-mini
            |--v1.0-test
            |--v1.0-trainval
            |--LICENSE
            |--nuscenes_infos_10sweeps_train.pkl
            |--nuscenes_infos_10sweeps_val.pkl

        ROOT_PATH = "Your Path/nuscenes/v1.0-trainval"
        DESCRIPTION = "nuscenes_infos_10sweeps_description.pkl"
    '''

    print(f"---- 开始合并 : nuscenes_infos_10sweeps_train.pkl和nuscenes_infos_10sweeps_val.pkl ----")
    generate_infos_train_val_pkl()
    print(f"---- 结束合并 : nuscenes_infos_10sweeps_train.pkl和nuscenes_infos_10sweeps_val.pkl ----")

    print(f"---- 开始获取原始nuScenes的Semantci和Description ----")
    get_semantic_and_description()
    print(f"---- 结束获取原始nuScenes的Semantci和Description ----")

    print(f"---- 开始生成不同类型的Text Description ----")
    get_text_decription()
    print(f"---- 结束生成不同类型的Text Description ----")

    print_info()
# ---- 一键生成 ----

if __name__ == '__main__':

    generate_descirption_pkl()
    pass

