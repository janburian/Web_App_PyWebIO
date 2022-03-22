import os.path
from pathlib import Path
import sys
import skimage.io
import numpy as np
import math
import matplotlib.pyplot as plt

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image
from datetime import date


def get_image_properties(dataset_directory):
    """
    Returns the properties of the images (list of dictionaries) which are obligatory in COCO data format
    For example (properties of one image):
        image {
            "id": int,
            "width": int,
            "height": int,
            "file_name": str,
        }

    Parameters:
    -----------
    dataset_directory : Path
                        directory of the image dataset
    """
    image_name = 0
    list_image_dictionaries = []
    image_name_id = 0

    while True:
        filename_string = (
            str(dataset_directory) + "\\" + str(image_name).zfill(4) + ".jpg"
        )
        filename_path = Path(filename_string)
        if not filename_path.exists():
            break
        image = skimage.io.imread(filename_string)
        height = image.shape[0]
        width = image.shape[1]

        image_dictionary = {
            "id": image_name_id,
            "width": width,
            "height": height,
            "file_name": str(image_name).zfill(4) + ".jpg",
        }
        list_image_dictionaries.append(image_dictionary)

        image_name += 1
        image_name_id += 1
    return list_image_dictionaries


def get_category_properties(dataset_directory, filename):
    """
    Returns properties of category (list of dictionaries)
    It can read categories from .txt file, if each category is on a single line without commas
    For example (properties of one category):
        category {
            "id": int,
            "name": str,
            "supercategory": str,
        }

    Parameters:
    -----------
    dataset_directory : Path
                        directory of the image dataset
    filename : str
               filename of the .txt file which is in the same file as dataset
    """
    list_category_dictionaries = []
    with open(str(dataset_directory) + "\\" + filename) as f:
        lines = f.readlines()
    number_lines = len(lines)

    for i in range(number_lines):
        supercategory = lines[i].rstrip()
        category_dictionary = {
            "id": i + 1,
            "name": supercategory,
            "supercategory": supercategory,
        }
        list_category_dictionaries.append(category_dictionary)

    return list_category_dictionaries


def count_polygon_area(x, y):
    """
    Counts the area of an polygon

    Parameters:
    -----------
    x : np.array(x_px_list),
    y : np.array(y_px_list)
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_annotations_properties(czi_files_directory, annotation_name, pixelsize_mm):
    """
    Returns the properties of the annotations (list of dictionaries)
    One dictionary = object instance annotation
    For example (one annotation):
        annotation{
            "id": int,
            "image_id": int, (the ID of a picture where the annotation is located)
            "category_id": int,
            "segmentation": RLE or [polygon],
            "area": float,
            "bbox": [x,y,width,height],
            "iscrowd": 0 or 1,
        }

    Parameters:
    -----------
    czi_files_directory : Path
                          path to .czi files directory
    annotation_name : Str
                      it is supposed, that the all annotations have the same name (e.g. annotation0001, annotation0002,...)
    pixelsize_mm : list
                   defines pixelsize in mm (e.g. pixelsize = [[0.0003, 0.0003])

    """

    index = 0
    annotation_id = 1
    category_id = 1  # only one category - cells
    image_id = 0

    list_annotation_dictionaries = []

    while True:
        filename_string = (
            str(czi_files_directory)
            + "\\"
            + annotation_name
            + str(index).zfill(4)
            + ".czi"
        )
        filename_path = Path(filename_string)
        if not filename_path.exists():
            break

        anim = scaffan.image.AnnotatedImage(path=str(filename_path))
        view = anim.get_full_view(
            pixelsize_mm=[0.0003, 0.0003]
        )  # wanted pixelsize in mm in view
        annotations = view.annotations

        """ 
        img = view.get_raster_image()
        view.plot_annotations()
        plt.imshow(img)
        plt.show()
        """

        for j in range(len(annotations)):
            xy_px_list = []

            x_px_list = (np.asarray(annotations[j]["x_mm"]) / pixelsize_mm[0]).tolist()
            y_px_list = (np.asarray(annotations[j]["y_mm"]) / pixelsize_mm[1]).tolist()

            x_px_min = float(math.floor(np.min(x_px_list)))
            y_px_min = float(math.floor(np.min(y_px_list)))
            width = float(math.floor(np.max(x_px_list)) - x_px_min)
            height = float(math.floor(np.max(y_px_list)) - y_px_min)

            bbox = [x_px_min, y_px_min, width, height]

            # polygon_area = count_polygon_area(np.array(x_px_list) * 0.0003, np.array(y_px_list) * 0.0003) # in mm
            polygon_area = count_polygon_area(
                np.array(x_px_list), np.array(y_px_list)
            )  # in pixels
            x_px_list = np.asarray(x_px_list)
            for i in range(len(x_px_list)):
                xy_px_list.append(x_px_list[i])
                xy_px_list.append(y_px_list[i])

            segmentation = xy_px_list

            # segmentation[0::2]
            # segmentation[1::2]
            # np.max(x_px_list)

            annotation_dictionary = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],  # RLE or [polygon]
                "area": polygon_area,  # Shoelace formula
                "bbox": bbox,  # [x, y, width, height]
                "iscrowd": 0,
            }
            annotation_id += 1

            list_annotation_dictionaries.append(annotation_dictionary)

        image_id += 1
        index += 1

    return list_annotation_dictionaries


def get_info_dictionary(version, description, contributor):
    """
    Returns dictionary with the basic information related to COCO dataset

    Parameters:
    -----------
    version : str
    description: str
    contributor : st
    """

    info_dictionary = {
        "year": str(date.today().year),
        "version": version,
        "description": description,
        "contributor": contributor,
        "date_created": date.today().strftime("%d/%m/%Y"),
    }
    return info_dictionary


if __name__ == "__main__":
    """
    Creating of .json file
    """
    import json

    """
    .json file structure
    
    data = {    
        "info" : info, 
        "images": [image],
        "categories": [],
        "annotations": [annotation]
        "licenses": [license]
    }
    """

    """
    PARAMETERS 
    """
    # Directory of the image dataset
    dataset_directory = Path(
        r"H:\BP\datasets\dataset_final\COCO_dataset_final_validate\images"
    )

    # Directory of the .czi files
    czi_files_directory = Path(
        r"H:\BP\data\dataset_final\czi_files_validate"
    )  # path to .czi files directory

    path_json = Path(
        r"H:\BP\datasets\dataset_final\COCO_dataset_final_validate"
    )  # path to directory, where the .json file will be saved
    name_json = "trainval.json"

    data = {}

    """
    Info
    """
    version = "1.0"
    description = "COCO dataset for scaffan"
    contributor = "Jan Burian"

    info_dictionary = get_info_dictionary(version, description, contributor)
    data.update({"info": info_dictionary})

    """
    Images
    """
    list_image_dictionaries = get_image_properties(dataset_directory)
    data.update({"images": list_image_dictionaries})

    """
    Categories
    """
    list_category_dictionaries = get_category_properties(
        dataset_directory, "categories.txt"
    )  # directory and .txt file
    data.update({"categories": list_category_dictionaries})

    """
    Annotations
    """
    annotation_name = "annotation"
    pixelsize_mm = [0.0003, 0.0003]
    list_annotation_dictionaries = get_annotations_properties(
        czi_files_directory, annotation_name, pixelsize_mm
    )
    data.update({"annotations": list_annotation_dictionaries})

    """
    Final .json file
    """
    # Creating .json file
    with open(os.path.join(path_json, name_json), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
