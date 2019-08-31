import os
from glob import glob
import csv

from ncc.readers.search_xml_profile import get_classname, convert_annotation


def list_detection_files(data_dir_path, image_dir, xml_dir, classes):
    """
    One row for one image;
    Row format: image_file_path box1 box2 ... boxN
    Box format: x_min,y_min,x_max,y_max,class_id
    :param data_dir: data_dir/(images or xmls)/image_file
    :param image_dir
    :param xml_dir
    """
    IMAGE_EXTENTINS = ['.jpg', '.png', '.JPG']
    image_dir_path = os.path.join(data_dir_path, image_dir)
    xml_dir_path = os.path.join(data_dir_path, xml_dir)
    xml_files = glob(os.path.join(xml_dir_path, '*.xml'))

    annotation_set = list()
    for xml_path in xml_files:
        xml_annotation = convert_annotation(xml_path, classes)
        xml_file_name = os.path.basename(xml_path)
        xml_no_ex_path, ex = os.path.splitext(xml_file_name)
        image_no_ex_path = os.path.join(image_dir_path, xml_no_ex_path)
        for image_ex in IMAGE_EXTENTINS:
            image_path = image_no_ex_path + image_ex
            if os.path.exists(image_path):
                annotation_set.append(image_path + xml_annotation)

    return annotation_set


def read_classification_set(target_path, class_names):
    IMAGE_EXTENTINS = ['.jpg', '.png', '.JPG']

    annotations = list()
    for class_id, class_name in enumerate(class_names):
        image_paths = list()
        class_dir_paths = [
            os.path.join(target_path,  class_name),
            os.path.join(target_path, '*',  class_name)
        ]
        for class_dir_path in class_dir_paths:
            for image_ex in IMAGE_EXTENTINS:
                image_paths += glob(
                    os.path.join(class_dir_path, '*', image_ex)
                )
        class_annotation = [
            [image_path, class_id] for image_path in image_paths
        ]
        annotations.extend(class_annotation)

    return annotations


def read_segmentation_set(target_path, image_dir, mask_dir):
    IMAGE_EXTENTINS = ['.jpg', '.png', '.JPG']

    annotations = list()
    image_dir_paths = [
        os.path.join(target_path, image_dir),
        os.path.join(target_path, '*', image_dir)
    ]
    mask_dir_paths = [
        os.path.join(target_path, mask_dir),
        os.path.join(target_path, '*', mask_dir)
    ]
    for image_dir_path, mask_dir_path in zip(image_dir_paths, mask_dir_paths):
        mask_paths = list()
        for image_ex in IMAGE_EXTENTINS:
            mask_paths += glob(
                os.path.join(mask_dir_path, '*', image_ex)
            )
        for mask_path in mask_paths:
            file_name = os.path.splitext(os.path.basename(mask_path))
            for image_ex in IMAGE_EXTENTINS:
                image_path = os.path.join(
                    image_dir_path, file_name, image_ex
                )
                if os.path.exists(image_path):
                    annotations.append([image_path, mask_path])

    return annotations


def detection_set(target_dir, input_dirs, image_dir, xml_dir,
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
        if os.path.isdir(data_dir_path) and data_dir in input_dirs:
            if training:
                xml_dir_path = os.path.join(data_dir_path, xml_dir)
                xml_files = glob(os.path.join(xml_dir_path, '*.xml'))
                class_names += get_classname(xml_files)

    # set class name and get class id.
    if training:
        class_names = list(set(class_names))

    for data_dir in data_dirs:
        data_dir_path = os.path.join(target_dir, data_dir)
        if os.path.isdir(data_dir_path) and data_dir in input_dirs:
            data_set += list_detection_files(
                data_dir_path, image_dir, xml_dir, class_names)

    return data_set, class_names


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
