# coding=utf-8
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

ROOT_PATH = "/ihoment/youjie10/qwt/dataset/nuscenes/v1.0-trainval"
INFO_PATH = f"{ROOT_PATH}/nuscenes_dbinfos_10sweeps_withvelo.pkl"
NUM_THRESHOLD = 5

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

def orientation_text_rad(boxes):
    """
    将车辆 yaw（弧度制）转换为前/后/左/右朝向描述
    """
    yaw_rad = boxes[6]
    yaw_deg = math.degrees(yaw_rad) % 360  # 转为 0~360 度

    if 315 <= yaw_deg or yaw_deg < 45:
        return "facing forward"
    elif 45 <= yaw_deg < 135:
        return "facing left"
    elif 135 <= yaw_deg < 225:
        return "facing backward"
    else:
        return "facing right"

def orientation_text_rad2(boxes):
    """
    将车辆 yaw（弧度制）转换为前/后/左/右朝向描述
    """
    yaw_rad = boxes[6]
    yaw_deg = math.degrees(yaw_rad) % 360  # 转为 0~360 度

    if 315 <= yaw_deg or yaw_deg < 135:
        return "facing forward"
    else:
        return "facing backward"

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


def count_num(gt_names):
    counter = Counter(gt_names)
    return counter


def generate_description(infos):
    num = len(infos)

    for i, info in enumerate(infos):
        print(f"----- {i}/{num} -----")
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


def save_pkl(save_path=None, infos=None):
    if (save_path is None):
        save_path = f"{ROOT_PATH}/nuscenes_infos_10sweeps_description.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
        print(f"---- Saving : {save_path} ----")


def generate_description_pkl():
    train_infos = generate_description(read_nuscenes_infos_10sweeps("train"))
    val_infos = generate_description(read_nuscenes_infos_10sweeps("val"))

    # infos = []
    # infos.append(train_infos)
    # infos.append(val_infos)

    infos = train_infos + val_infos

    save_path = f"{ROOT_PATH}/nuscenes_infos_10sweeps_description.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
        print(f"---- Saving : {save_path} ----")


def sort_number_word_phrases(phrases):
    """
    将类似 ["one car", "nine cars", "three cars"] 的列表按数字大小排序
    """
    return sorted(
        phrases,
        key=lambda p: w2n.word_to_num(p.split()[0])  # 提取第一个单词作为数字
    )


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


def read_pkl(
    file_path = f"{ROOT_PATH}/nuscenes_infos_10sweeps_description.pkl"
):
    with open(file_path, 'rb') as f:
        infos = pickle.load(f)

    return infos


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

def text_aim(item, object="car", threshold=NUM_THRESHOLD):
    # Exmaple : Two cars.

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num <= threshold):
        text = f"Less than {num2words(threshold)} {object}{suffix}."
    else:
        text = f"More than {num2words(threshold)} {object}{suffix}."
    return text


def text_l1(item, object="car", threshold=NUM_THRESHOLD):
    # Exmaple : Two cars.

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num == 0):
        text = f"No {object}."
    elif (car_num == 1):
        text = f"One {object}."
    elif (car_num <= threshold):
        text = f"{car_num_word} {object}{suffix}."
        text = text.capitalize()
    else:
        text = f"More than {num2words(threshold)} {object}{suffix}."
    return text


def text_l2(item, object="car", threshold=NUM_THRESHOLD):
    # Exmaple : There are two cars in the scene.

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num == 0):
        text = f"There is no {object} in the scene."
    elif (car_num == 1):
        text = f"There is {car_num_word} {object} in the scene."
    elif (car_num <= threshold):
        text = f"There are {car_num_word} {object}{suffix} in the scene."
    else:
        text = f"There are more than {num2words(threshold)} {object}{suffix} in the scene."
    return text


def text_l3(item, object="car", threshold=NUM_THRESHOLD):
    # Exmaple : There are two cars in the scene. One car is in front. One car is behind.

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num == 0):
        text = f""
    elif (car_num == 1 and car_num <= threshold):
        text = f"One {object} is in front."
    elif (car_num == 2 and car_num <= threshold):
        text = f"One {object} is in front. One {object} is behind."
    elif (car_num == 3 and car_num <= threshold):
        text = f"One {object} is in front. Two {object}{suffix} are behind."
    elif (car_num == 4 and car_num <= threshold):
        text = f"Two {object}{suffix} are in front. Two {object}{suffix} are behind."
    elif (car_num == 5 and car_num <= threshold):
        text = f"Two {object}{suffix} are in front. Three {object}{suffix} are behind."
    else:
        text = f"One {object} is in front. Other {object}{suffix} are behind."
    return text


# def text_l3(item, object="car", threshhold=NUM_THRESHOLD):
#     gt_names = item["gt_names"]
#     gt_boxes = item["gt_boxes"]
#
#     if (not object in gt_names.tolist()):
#         return f"There is no {object} in the scene."
#
#     all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)
#
#     car_num = len(object_boxes)
#     car_num_word = num2words(car_num)
#
#     suffix = "s"
#     if (object == "bus"):
#         suffix = "es"
#
#     text = ""
#     if (car_num == 1):
#         text = f"There is {car_num_word} {object} in the scene."
#         if (len(all_descs) > 0):
#             text2 = f"{car_num_word} {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}"
#             text2 = text2.capitalize()
#             text = text + " " + text2
#
#     elif (car_num == 2):
#         text = f"There are {car_num_word} {object}{suffix} in the scene."
#         if (len(all_descs) > 0):
#             text2 = f"One {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}."
#             text = text + " " + text2
#
#             if (len(all_descs) > 1):
#                 text3 = f"One {object} is {all_descs[1][1][0]} to the {all_descs[1][1][1]} of one {other_names[1]}."
#                 text = text + " " + text3
#             else:
#                 text3 = f"One {object} is {all_descs[0][1][0]} to the {all_descs[0][1][1]} of one {other_names[0]}."
#                 text = text + " " + text3
#
#     elif (car_num >= threshhold):
#         if (car_num > threshhold):
#             text = f"There are more than {num2words((threshhold))} {object}{suffix} in the scene."
#         else:
#             text = f"There are {car_num_word} {object}{suffix} in the scene."
#
#         max_i, max_indices, max_i2, max_indices2 = search_same_name(copy.deepcopy(all_descs))
#         if (len(all_descs) > 0):
#             if (len(max_indices) == 1):
#                 text2 = f"One {object} is {all_descs[max_i][max_indices[0]][0]} to the {all_descs[max_i][max_indices[0]][1]} of one {other_names[max_i]}."
#             elif (len(max_indices) < threshhold):
#                 text2 = f"{num2words(len(max_indices))} {object}{suffix} are {all_descs[max_i][max_indices[0]][0]} to the {all_descs[max_i][max_indices[0]][1]} of one {other_names[max_i]}."
#                 text2 = text2.capitalize()
#             else:
#                 text2 = f"More than {num2words(threshhold)} {object}{suffix} are {all_descs[max_i][max_indices[0]][0]} to the {all_descs[max_i][max_indices[0]][1]} of one {other_names[max_i]}."
#             text = text + " " + text2
#
#             if (len(max_indices2) > 0):
#                 if (len(max_indices2) == 1):
#                     text3 = f"One {object} is {all_descs[max_i2][max_indices2[0]][0]} to the {all_descs[max_i2][max_indices2[0]][1]} of one {other_names[max_i2]}."
#                 elif (len(max_indices2) < threshhold):
#                     text3 = f"{num2words(len(max_indices2))} {object}{suffix} are {all_descs[max_i2][max_indices2[0]][0]} to the {all_descs[max_i2][max_indices2[0]][1]} of one {other_names[max_i2]}."
#                     text3 = text3.capitalize()
#                 else:
#                     text3 = f"More than {num2words(threshhold)} {object}{suffix} are {all_descs[max_i2][max_indices2[0]][0]} to the {all_descs[max_i2][max_indices2[0]][1]} of one {other_names[max_i2]}."
#                 text = text + " " + text3
#     return text


# def text_l4(item, object="car", threshhold=NUM_THRESHOLD):
#     # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone. One car is ahead to the right of one traffic_cone.
#
#     gt_names = item["gt_names"]
#     gt_boxes = item["gt_boxes"]
#
#     if (not object in gt_names.tolist()):
#         #return f"There is no {object} in the scene."
#         return ""
#
#     all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)
#
#     car_num = len(object_boxes)
#     car_num_word = num2words(car_num)
#
#     suffix = "s"
#     if (object == "bus"):
#         suffix = "es"
#
#     text = ""
#     if (car_num == 1):
#         # text = f"There is {car_num_word} {object} in the scene."
#         if (len(all_descs) > 0):
#             text2 = f"{car_num_word} {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}."
#             text2 = text2.capitalize()
#             text = text + " " + text2
#
#     elif (car_num >= threshhold):
#         # if (car_num > threshhold):
#         #     text = f"There are more than {num2words((threshhold))} {object}{suffix} in the scene."
#         # else:
#         #     text = f"There are {car_num_word} {object}{suffix} in the scene."
#
#         if (len(all_descs) > 0):
#             text2 = f"One {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}."
#             text = text + " " + text2
#
#         if (len(all_descs) > 1):
#             text3 = f"One {object} is {all_descs[1][1][0]} to the {all_descs[1][1][1]} of one {other_names[1]}."
#             text = text + " " + text3
#
#     return text

def text_l4(item, object="car", threshhold=NUM_THRESHOLD):
    # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone. One car is ahead to the right of one traffic_cone.

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    if (not object in gt_names.tolist()):
        #return f"There is no {object} in the scene."
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}."

    return text


def text_l5(item, object="car", threshhold=NUM_THRESHOLD):
    # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone. One car is ahead to the right of one traffic_cone.

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    if (not object in gt_names.tolist()):
        #return f"There is no {object} in the scene."
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is {all_descs[0][0][0]} to one {other_names[0]}."

    return text

def text_l6(item, object="car", threshhold=NUM_THRESHOLD):
    # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone. One car is ahead to the right of one traffic_cone.

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    if (not object in gt_names.tolist()):
        #return f"There is no {object} in the scene."
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is the {all_descs[0][0][1]} of one {other_names[0]}."

    return text

def text_l7(
        item,
        object="car",
        threshhold=NUM_THRESHOLD,
        keys = ["pedestrian", "barrier", "truck"]
):
    # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone. One car is ahead to the right of one traffic_cone.

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    if (not object in gt_names.tolist()):
        #return f"There is no {object} in the scene."
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    text = ""

    if (len(all_descs) > 0):
        if(keys.__contains__(other_names[0])):
            text = f"One {object} is around one {other_names[0]}."

    return text

def text_l8(item, object="car", threshhold=NUM_THRESHOLD):
    # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone, facing forward. One car is ahead to the right of one traffic_cone, facing left.

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    if (not object in gt_names.tolist()):
        # return f"There is no {object} in the scene."
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is {orientation_text_rad(other_boxes[0])}."


    return text

def text_l9(item, object="car", threshhold=NUM_THRESHOLD):
    # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone, facing forward. One car is ahead to the right of one traffic_cone, facing left.

    gt_names = item["gt_names"]
    gt_boxes = item["gt_boxes"]

    if (not object in gt_names.tolist()):
        # return f"There is no {object} in the scene."
        return ""

    all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)

    car_num = len(object_boxes)
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    text = ""

    if (len(all_descs) > 0):
        text = f"One {object} is {orientation_text_rad2(other_boxes[0])}."


    return text

# def text_l6(item, object="car", threshhold=NUM_THRESHOLD):
#     # Example: There are two cars in the scene. One car is ahead to the right of one traffic_cone, facing forward. One car is ahead to the right of one traffic_cone, facing left.
#
#     gt_names = item["gt_names"]
#     gt_boxes = item["gt_boxes"]
#
#     if (not object in gt_names.tolist()):
#         # return f"There is no {object} in the scene."
#         return ""
#
#     all_descs, other_boxes, other_names, object_boxes = get_boxes_to_boxes(item, object)
#
#     car_num = len(object_boxes)
#     car_num_word = num2words(car_num)
#
#     suffix = "s"
#     if (object == "bus"):
#         suffix = "es"
#
#     text = ""
#
#     if (len(all_descs) > 0):
#         text2 = f"{car_num_word} {object} is {all_descs[0][0][0]} to the {all_descs[0][0][1]} of one {other_names[0]}, {orientation_text_rad(other_boxes[0])}."
#         text2 = text2.capitalize()
#         text = text + text2
#
#     return text

def generate_l1_to_l4():
    '''
        文本 -> 点云场景中类别和目标数量控制
        文本 -> 点云场景上采样
        文本 -> 点云场景下采样
        文本 -> 点云场景天气生成？

        对多不超过5辆车
        L1:
            There are X cars in the scene.

        L2:
            There are X cars in the scene. X cars ahead, X cars behind.

        L3:
            There are X cars in the scene. X cars are in front of the bus, ...

        L4:
            There are X cars in the scene. X cars are in front of the bus, facing ..., ...

    '''
    data = f"{ROOT_PATH}/nuscenes_infos_10sweeps_description.pkl"
    nuscenes_infos = []
    with open(data, 'rb') as f:
        infos = pickle.load(f)
        nuscenes_infos.extend(infos)

    all = nuscenes_infos

    object = "car"

    for i, item in enumerate(all):
        print(f"---- {i}/{len(all)} ----")
        item["text_l0"] = text_l0(item, object)
        item["text_l1"] = text_l1(item, object)
        item["text_l2"] = text_l2(item, object)
        item["text_l3"] = text_l3(item, object)
        item["text_l4"] = text_l4(item, object)
        all[i] = item

    save_pkl(save_path=data, infos=all)


def get_point_cloud(lidar_path):
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
    return points


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


def generate_class_boxes():
    data = f"{ROOT_PATH}/nuscenes_infos_10sweeps_description.pkl"
    nuscenes_infos = []
    with open(data, 'rb') as f:
        infos = pickle.load(f)
        nuscenes_infos.extend(infos)

    all = nuscenes_infos

    for i in range(len(all)):
        print(f"---- {i}/{len(all)} ----")
        item = all[i]
        gt_boxes = get_class_label(item["gt_names"], item["gt_boxes"])
        item["gt_boxes_class"] = gt_boxes
        all[i] = item

    save_pkl(data, infos)

    return all



def point_in_boxes(point, boxes):
    """
    point: [3] (x,y,z)
    boxes: [M,10], center=(0:3), size=(3:6), yaw=6
    return: inside_mask [M] bool
    """
    device = boxes.device
    p = point.to(device).float()  # [3]

    centers = boxes[:, 0:3]  # [M,3]
    sizes = boxes[:, 3:6]  # [M,3] -> (dx,dy,dz)
    yaw = boxes[:, 6]  # [M]

    # 旋转矩阵 Rz(yaw)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    R = torch.zeros((boxes.shape[0], 3, 3), device=device)
    R[:, 0, 0] = cos_y;
    R[:, 0, 1] = -sin_y
    R[:, 1, 0] = sin_y;
    R[:, 1, 1] = cos_y
    R[:, 2, 2] = 1.0

    # 点相对盒子中心
    p_rel = p.unsqueeze(0) - centers  # [M,3]
    # 转到盒子局部坐标
    p_local = torch.einsum('mij,mj->mi', R.transpose(1, 2), p_rel)  # [M,3]

    # 判断在盒子内
    half = sizes / 2.0
    inside_x = (p_local[:, 0] >= -half[:, 0]) & (p_local[:, 0] <= half[:, 0])
    inside_y = (p_local[:, 1] >= -half[:, 1]) & (p_local[:, 1] <= half[:, 1])
    inside_z = (p_local[:, 2] >= -half[:, 2]) & (p_local[:, 2] <= half[:, 2])

    inside_mask = inside_x & inside_y & inside_z  # [M]
    return inside_mask


def semantic_labels_from_boxes(
        points: torch.Tensor,  # [N,3], float32/float16, LiDAR坐标系
        boxes: torch.Tensor,  # [M,10], (cx,cy,cz, dx,dy,dz, yaw, ..., class_id)
        background_id: int = 10,
        ignore_ids=None,  # e.g. {255}；None 表示不过滤
        prefer: str = "nearest_center",  # or "smallest_box"
        box_chunk: int = None  # e.g. 256/512；None=不分块
):
    """
    返回:
      sem_labels: [N] (torch.long)  每个点的语义类别（未命中=background_id）
    说明:
      - yaw 视为绕 +Z 轴的右手旋转（弧度）
      - 多盒命中用 prefer 决策："nearest_center" 或 "smallest_box"
      - 可选 box_chunk 对 M 分块，降低显存
    """
    device, dtype = points.device, points.dtype
    N = points.shape[0]
    M = boxes.shape[0]

    # 默认结果：全背景
    sem_labels = torch.full((N,), background_id, dtype=torch.long, device=device)
    if N == 0 or M == 0:
        return sem_labels

    centers_all = boxes[:, 0:3].to(device=device, dtype=dtype)  # [M,3]
    sizes_all = boxes[:, 3:6].to(device=device, dtype=dtype)  # [M,3] (dx,dy,dz)
    yaw_all = boxes[:, 6].to(device=device, dtype=dtype)  # [M]
    cls_all = boxes[:, 9].long().to(device)  # [M]

    # 过滤 ignore 类（可选）
    if ignore_ids is not None and len(ignore_ids) > 0:
        keep = ~torch.isin(cls_all, torch.as_tensor(list(ignore_ids), device=device))
        centers_all, sizes_all, yaw_all, cls_all = centers_all[keep], sizes_all[keep], yaw_all[keep], cls_all[keep]
        M = centers_all.shape[0]
        if M == 0:
            return sem_labels

    # 维护全局最优（每个点）
    if prefer == "nearest_center":
        best_score = torch.full((N,), float("inf"), dtype=dtype, device=device)  # 最小中心距离^2
    elif prefer == "smallest_box":
        best_score = torch.full((N,), float("inf"), dtype=dtype, device=device)  # 最小体积
    else:
        raise ValueError(f"Unknown prefer={prefer}")

    best_cls = torch.full((N,), background_id, dtype=torch.long, device=device)

    # 分块遍历盒子，避免一次性 M×N 太大
    if box_chunk is None or box_chunk <= 0:
        box_chunk = M

    for s in range(0, M, box_chunk):
        e = min(s + box_chunk, M)
        centers = centers_all[s:e]  # [m,3]
        sizes = sizes_all[s:e]  # [m,3]
        yaw = yaw_all[s:e]  # [m]
        cls_id = cls_all[s:e]  # [m]
        m = centers.shape[0]
        if m == 0:
            continue

        # Rz(yaw)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)  # [m]
        R = torch.zeros((m, 3, 3), dtype=dtype, device=device)
        R[:, 0, 0] = cos_y;
        R[:, 0, 1] = -sin_y
        R[:, 1, 0] = sin_y;
        R[:, 1, 1] = cos_y
        R[:, 2, 2] = 1.0

        # p_local = R^T * (p - c) ；广播到 [m,N,3]
        p_rel = points.unsqueeze(0) - centers.unsqueeze(1)  # [m,N,3]
        p_local = torch.einsum('mij,mnj->mni', R.transpose(1, 2), p_rel)  # [m,N,3]

        half = sizes / 2.0
        inside = (p_local[:, :, 0].abs() <= half[:, 0, None]) \
                 & (p_local[:, :, 1].abs() <= half[:, 1, None]) \
                 & (p_local[:, :, 2].abs() <= half[:, 2, None])  # [m,N]

        if prefer == "nearest_center":
            # 分数=到盒子中心的距离^2，inside 之外设为 +inf
            dist2 = (p_rel ** 2).sum(dim=2)  # [m,N]
            score = torch.where(inside, dist2, torch.full_like(dist2, float("inf")))
        else:  # "smallest_box"
            vol = (sizes[:, 0] * sizes[:, 1] * sizes[:, 2]).unsqueeze(1)  # [m,1]
            score = torch.where(inside, vol, torch.full_like(vol, float("inf")))  # [m,N]

        # 在当前块内选最佳盒子
        cur_best_score, cur_best_idx = score.min(dim=0)  # [N], [N]
        improve = cur_best_score < best_score  # [N]

        # 仅在有命中(非 inf) 且 更好 时更新
        hit_and_better = improve & torch.isfinite(cur_best_score)

        best_score[hit_and_better] = cur_best_score[hit_and_better]
        best_cls[hit_and_better] = cls_id[cur_best_idx[hit_and_better]]

    sem_labels = best_cls  # [N]
    return sem_labels


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


def generate_senmantic_map():
    infos = read_pkl()
    for i, info in enumerate(infos):
        print(f"---- {i}/{len(infos)} ----")
        points = get_point_cloud(f"{ROOT_PATH}" + info["lidar_path"])

        points = torch.from_numpy(points).cuda()
        gt_boxes_class = torch.from_numpy(info["gt_boxes_class"]).cuda()
        semantic = semantic_labels_from_boxes(
            points=points,
            boxes=gt_boxes_class,
            background_id=gt_boxes_class.shape[-1]
        )

        info["semantic"] = semantic
        infos[i] = info

        # points = points.cpu().numpy()
        # semantic = semantic.int().cpu().numpy()

        # gt_boxes_class = info["gt_boxes_class"]
        # semantic = []
        # for j,point in enumerate(points):
        #     print(f"---- {j}/{len(points)} ----")
        #     fig = False
        #     for box in gt_boxes_class:
        #         fig = check_point_in_box(point, box)
        #         if(fig):
        #             label = box[-1]
        #             semantic.append(label)
        #             break
        #     if(not fig):
        #         semantic.append(10)
        # semantic = np.array(semantic).astype(np.int32)
        #
        #
        # colorize_point_by_label(points, semantic, f"{i*10}")

    save_pkl(infos=infos)

def generate_text_semantic():

    generate_l1_to_l4()
    generate_class_boxes()
    generate_senmantic_map()

def get_semantic_and_description():

    dataroot = ROOT_PATH
    version = 'v1.0-trainval'
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    infos = read_pkl()
    for i, info in enumerate(infos):
        print(f"----------  {i+1}/{len(infos)}  ----------")

        sample_token = info['token']
        sample = nusc.get("sample", sample_token)

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

        scene_token = sample["scene_token"]
        scene = nusc.get("scene", scene_token)
        description = scene["description"]
        info["description"] = description

        infos[i] = info

    save_pkl(infos=infos)


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


def check_weather(text):
    if(text.__contains__("rain") or text.__contains__("Rain")):
        return "Rainy."
    else:
        return "Sunny."

def check_time(text):
    if(text.__contains__("night") or text.__contains__("Night")):
        return "Night."
    else:
        return "Day."

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

def sorted_dict(info):
    keys = [
        "lidar_path",
        "cam_front_path",
        "cam_intrinsic",
        "token",
        "sweeps",
        "ref_from_car",
        "car_from_global",
        "timestamp",
        "location",
        "gt_boxes",
        "gt_boxes_token",
        "gt_boxes_class",
        "gt_boxes_velocity",
        "gt_names",
        "num_lidar_pts",
        "num_radar_pts",
        "description",
        "description_class",
        "semantic",
        "text_l0",
        "text_l1",
        "text_l2",
        "text_l3",
        "text_l4",
        "text_l5",
        "text_l6",
        "text_l7",
    ]
    new_info={}
    for key in keys:
        new_info[key] = info[key]

    return new_info

def print_dict(infos):

    for k,v in infos.items():
        print(f"k : {k}, v : {infos[k]}")

def del_key(info, key):

    del info[key]
    return info

def final_dec():
    textaim = {}
    textl1 = {}
    textl2 = {}
    textl3 = {}
    textl4 = {}
    textl5 = {}
    textl6 = {}
    textl7 = {}
    textl8 = {}
    textl9 = {}
    textl10 = {}
    textl11 = {}

    infos = read_pkl()
    for i, info in enumerate(infos):
        print(f"---- {i}/{len(infos)} ----")

        # info = del_key(info, "text_l0")
        # info = del_key(info, "text_l1")
        # info = del_key(info, "text_l2")
        # info = del_key(info, "text_l3")
        # info = del_key(info, "text_l4")
        # info = del_key(info, "text_l5")
        # info = del_key(info, "text_l6")
        # info = del_key(info, "text_l7")
        # info = del_key(info, "text_l8")
        # info = del_key(info, "text_l9")

        # ---- text_l1 ----
        info["text_l1"] = text_l1(info)
        print(info["text_l1"])

        t1 = info["text_l1"]
        if(not textl1.keys().__contains__(t1)):
            textl1[t1] = 1
        else:
            textl1[t1] += 1
        info["text_l1_num"] = textl1
        # ---- text_l1 ----

        # ---- text_l2 ----
        info["text_l2"] = text_l2(info)
        print(info["text_l2"])

        t2 = info["text_l2"]
        if(not textl2.keys().__contains__(t2)):
            textl2[t2] = 1
        else:
            textl2[t2] += 1
        info["text_l2_num"] = textl2
        # ---- text_l2 ----

        # ---- text_l3 ----
        info["text_l3"] = text_l3(info)
        print(info["text_l3"])

        t3 = info["text_l3"]
        if(not textl3.keys().__contains__(t3)):
            textl3[t3] = 1
        else:
            textl3[t3] += 1
        info["text_l3_num"] = textl3
        # ---- text_l3 ----

        # ---- text_l4 ----
        info["text_l4"] = text_l4(info)
        print(info["text_l4"])

        t4 = info["text_l4"]
        if(not textl4.keys().__contains__(t4)):
            textl4[t4] = 1
        else:
            textl4[t4] += 1
        info["text_l4_num"] = textl4
        # ---- text_l4 ----

        # ---- text_l5 ----
        info["text_l5"] = text_l5(info)
        print(info["text_l5"])

        t5 = info["text_l5"]
        if(not textl5.keys().__contains__(t5)):
            textl5[t5] = 1
        else:
            textl5[t5] += 1
        info["text_l5_num"] = textl5
        # ---- text_l5 ----

        # ---- text_l6 ----
        info["text_l6"] = text_l6(info)
        print(info["text_l6"])

        t6 = info["text_l6"]
        if(not textl6.keys().__contains__(t6)):
            textl6[t6] = 1
        else:
            textl6[t6] += 1
        info["text_l6_num"] = textl6
        # ---- text_l6 ----

        # ---- text_l7 ----
        info["text_l7"] = text_l7(info)
        print(info["text_l7"])

        t7 = info["text_l7"]
        if(not textl7.keys().__contains__(t7)):
            textl7[t7] = 1
        else:
            textl7[t7] += 1
        info["text_l7_num"] = textl7
        # ---- text_l7 ----

        # ---- text_l8 ----
        info["text_l8"] = text_l8(info)
        print(info["text_l8"])

        t8 = info["text_l8"]
        if(not textl8.keys().__contains__(t8)):
            textl8[t8] = 1
        else:
            textl8[t8] += 1
        info["text_l8_num"] = textl8
        # ---- text_l8 ----

        # ---- text_l9 ----
        info["text_l9"] = text_l9(info)
        print(info["text_l9"])

        t9 = info["text_l9"]
        if(not textl9.keys().__contains__(t9)):
            textl9[t9] = 1
        else:
            textl9[t9] += 1
        info["text_l9_num"] = textl9
        # ---- text_l9 ----

        # ---- text_l10 ----
        info["text_l10"] = check_weather(info["description"])
        print(info["text_l10"])

        t10 = info["text_l10"]
        if(not textl10.keys().__contains__(t10)):
            textl10[t10] = 1
        else:
            textl10[t10] += 1
        info["text_l10_num"] = textl10
        # ---- text_l10 ----

        # ---- text_l11 ----
        info["text_l11"] = check_time(info["description"])
        print(info["text_l11"])

        t11 = info["text_l11"]
        if(not textl11.keys().__contains__(t11)):
            textl11[t11] = 1
        else:
            textl11[t11] += 1
        info["text_l11_num"] = textl11
        # ---- text_l11 ----

        # ---- text_l0 ----
        info["text_l0"] = text_aim(info)
        print(info["text_l0"])

        t0 = info["text_l0"]
        if(not textaim.keys().__contains__(t0)):
            textaim[t0] = 1
        else:
            textaim[t0] += 1
        info["text_l0_num"] = textaim
        # ---- text_l0----

        # ---- text_aim ----
        l7 = info["text_l7"]
        if(l7 == ""):
            info["text_aim"] = info["text_l10"]
        else:
            info["text_aim"] = info["text_l10"] + " " + info["text_l7"]
        print(info["text_aim"])

        t0 = info["text_aim"]

        if(not textaim.keys().__contains__(t0)):
            textaim[t0] = 1
        else:
            textaim[t0] += 1
        info["text_aim_num"] = textaim
        # ---- text_aim----

        infos[i] = info

    save_pkl(infos=infos)

    print_dict(infos[0]["text_l0_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l1_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l2_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l3_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l4_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l5_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l6_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l7_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l8_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l9_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l10_num"])

    print("---- End ----")

    print_dict(infos[0]["text_l11_num"])

    print("---- End ----")

    print_dict(infos[0]["text_aim_num"])

def generate_pkl():
    generate_description_pkl()
    get_semantic_and_description()
    final_dec()

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

if __name__ == '__main__':
    '''
        文本 -> 点云场景中类别和目标数量控制
        文本 -> 点云场景上采样
        文本 -> 点云场景下采样
        文本 -> 点云场景天气生成？
        
        几何，天气，时间
        
        对多不超过5辆车
        L0:
            Less than five cars.
            
        L1:
            Two cars. ---!!! ----

        L2:
            There are two cars in the scene.

        L3:
            One car is in front. One car is behind.

        L4:
            One car is ahead to the right of one traffic_cone.
        
        L5:
            One car is ahead to one traffic_cone.
        
        L6:
            One car is the right of one traffic_cone.
        
        L7:
            One car is around one traffic_cone. ---!!! ----
            
        L8:
            One car is facing backward. ---!!! ----
            
        L9: 
            One car is facing backward.
            
        L10:
            Sunny
        
        L11:
            Day
    '''

    # textaim = {}
    #
    # infos = read_pkl()
    # for i, info in enumerate(infos):
    #     print(f"---- {i}/{len(infos)} ----")
    #
    #     # ---- text_aim ----
    #     text_l7 = info["text_l7"]
    #     text_l10 = info["text_l10"]
    #     if(text_l7 == ""):
    #         info["text_aim"] = info["text_l10"]
    #     else:
    #         info["text_aim"] = info["text_l10"] + " " + info["text_l7"]
    #     print(info["text_aim"])
    #
    #     t0 = info["text_aim"]
    #
    #     if(not textaim.keys().__contains__(t0)):
    #         textaim[t0] = 1
    #     else:
    #         textaim[t0] += 1
    #     info["text_aim_num"] = textaim
    #     # ---- text_aim----
    #
    #     infos[i] = info
    #
    # print_dict(infos[0]["text_aim_num"])
    #
    # save_pkl(infos=infos)

    # generate_description_pkl()
    # get_semantic_and_description()
    final_dec()

    # textl1 = {}
    #
    # infos = read_pkl()
    # for i, info in enumerate(infos):
    #     print(f"---- {i}/{len(infos)} ----")
    #
    #     # ---- text_l1 ----
    #     info["text_l1"] = text_l9(info)
    #     print(info["text_l1"])
    #
    #     t1 = info["text_l1"]
    #     if(not textl1.keys().__contains__(t1)):
    #         textl1[t1] = 1
    #     else:
    #         textl1[t1] += 1
    #     info["text_l1_num"] = textl1
    #     # ---- text_l1 ----
    #
    #     infos[i] = info
    #
    # print_dict(infos[0]["text_l1_num"])

    # infos = read_pkl()
    # for i, info in enumerate(infos):
    #     print(f"---- {i}/{len(infos)} ----")
    #
    #     info["text_l7"] = text_l7(info)
    #
    #     infos[i] = info
    #
    # save_pkl(infos=infos)

    # infos = read_pkl()
    #
    # count = {}
    # for info in infos:
    #     text = info["text_l11"] + info["text_l1"] + info["text_l7"] + info["text_l8"]
    #     # text = info["text_aim"]
    #     if(text not in count.keys()):
    #         count[text] = 1
    #     else:
    #         count[text] += 1
    #
    # for key in count.keys():
    #     print(key, count[key])
    # print(len(count.keys()))

    # generate_pkl()

    pass

