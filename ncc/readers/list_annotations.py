import os
from glob import glob
import csv


def list_classification_files(data_dir):
    """
    :param data_dir: this directory contains category directories
                    (data_dir/class_name/*.jpg)
    :return: [file path, class name]
    """
    image_files = list()
    labels = list()
    data_dir_list = os.listdir(data_dir)
    class_names = [class_name for class_name in data_dir_list if os.path.isdir(
        os.path.join(data_dir, class_name))]
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for image_ex in ['*.jpg', '*.png']:
            class_files = glob(os.path.join(class_dir, image_ex))
            image_files += class_files
            labels += [class_name for _ in range(len(class_files))]

    annotation_set = [[image_file, label]
                      for image_file, label in zip(image_files, labels)]

    return annotation_set, class_names


def list_segmentation_files(data_dir_path, image_dir, label_dir):
    """
    list raw image file path and mask file path
    :param data_dir: data_dir/(images or labels)/image_file
    :param image_dir
    :param label_dir
    :return: [image_file_path, label_file_path]
    """
    IMAGE_EXTENTINS = ['.jpg', '.png']
    label_files = list()
    image_dir_path = os.path.join(data_dir_path, image_dir)
    label_dir_path = os.path.join(data_dir_path, label_dir)
    for image_ex in IMAGE_EXTENTINS:
        image_path = "*" + image_ex
        label_files += glob(os.path.join(label_dir_path, image_path))

    annotation_set = list()
    for label_path in label_files:
        label_file_name = os.path.basename(label_path)
        label_no_ex_path, ex = os.path.splitext(label_file_name)
        image_no_ex_path = os.path.join(image_dir_path, label_no_ex_path)
        for image_ex in IMAGE_EXTENTINS:
            image_path = image_no_ex_path + image_ex
            if os.path.exists(image_path):
                annotation_set.append([image_path, label_path])

    return annotation_set


def classification_set(target_dir, input_dirs,
                       training=True, class_names=list()):
    """
    collect annotation files in target directory
    (target_dir/data_dir/class_dir/image_file)
    :param target_dir: root path that contains data set
    :param input_dirs: directory list used for train data
    :return: train_set: [image_file_path, label_idx]
             test_set: [image_file_path, label_idx]
    """
    data_set = list()
    data_dirs = os.listdir(target_dir)
    for data_dir in data_dirs:
        data_dir_path = os.path.join(target_dir, data_dir)
        if not os.path.isdir(data_dir_path):  # not a directory
            continue
        annotation_list, class_name_list = list_classification_files(
            data_dir_path)
        if data_dir in input_dirs:
            data_set += annotation_list
        if training:
            class_names += class_name_list

    # set class name and get class id.
    if training:
        class_names = list(set(class_names))
    data_set = [[path, class_names.index(name)] for path, name in data_set]

    return data_set, class_names


def segmentation_set(target_dir, input_dirs, image_dir, label_dir):
    """
    collect annotation files in target dir (target/data/class/image_file)
    :param target_dir: root path that contains data set
    :param input_dirs: directory list used for train/validation/test data
    :param image_dir: raw image directory name
    :param label_dir: mask image directory name
    :return: train_set: [image_file_path, label_file_path]
             test_set: [image_file_path, label_file_path]
    """
    data_set = list()
    data_dirs = os.listdir(target_dir)
    for data_dir in data_dirs:
        data_dir_path = os.path.join(target_dir, data_dir)
        annotation_list = list_segmentation_files(
            data_dir_path, image_dir, label_dir)
        if data_dir in input_dirs:
            data_set += annotation_list

    return data_set


def data_set_from_annotation(annotation_file):
    """
    collect annotation files in target dir (target/data/class/image_file)
    :param annotation_file: annotation_csv_file
    :return: train_set: [image_file_path, label_idx]
             test_set: [image_file_path, label_idx]
    """
    data_set = list()
    with open(annotation_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data_set.append(row)
    return data_set
