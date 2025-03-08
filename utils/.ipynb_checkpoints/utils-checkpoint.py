import pickle
import json
import random
import os
import numpy as np


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pkl_data(data, dir, file_name):
    create_dir(dir)
    pickle.dump(data, open(dir + file_name, 'wb'))


def load_pkl_data(dir, file_name):
    '''
    Args:
    -----
        path: path
        filename: file name
    Returns:
    --------
        data: loaded data
    '''
    file = open(dir+file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def save_json_data(data, dir, file_name):
    create_dir(dir)
    with open(dir+file_name, 'w') as fp:
        json.dump(data, fp)


import json
import os

def load_json_data(dir, file_name):
    """
    该函数用于加载指定目录下的 JSON 文件，并返回解析后的 JSON 数据。

    :param dir: 包含 JSON 文件的目录路径
    :param file_name: JSON 文件的名称
    :return: 解析后的 JSON 数据，如果出现错误则返回 None
    """
    # 拼接完整的文件路径
    file_path = os.path.join(dir, file_name)
    try:
        # 以只读模式打开 JSON 文件，并指定编码为 UTF-8
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用 json.load 方法将文件内容解析为 Python 对象
            data = json.load(file)
            return data
    except FileNotFoundError:
        # 处理文件未找到的异常
        print(f"文件 {file_path} 未找到。")
        return None

