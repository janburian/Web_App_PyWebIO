import os
import shutil
import sys

import pywebio.session
import skimage.io

from pywebio.input import *
from pywebio.output import *
import re

from pathlib import Path

import COCO_json

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image
import imma.image

import json

def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if (re.fullmatch(regex, email)):
        return
    return "Bad format of email."

def get_user_info():
    info = input_group("User info", [
      input('Input your name', name='name', required=True),
      input('Input your email', name='email', type=TEXT, required=True, validate=check_email)
    ])
    put_text(info['name'], info['email'])

    return info

def upload_data_page():
    data = input_group("Main page", [
        file_upload("Upload .czi files:", accept=".czi", multiple=True, required=True, name="imgs"),
        radio("Choose one option", options=['Predict', 'Train'], required=True, name="operation"),
    ])

    return data

def create_directory(directory_name):
    current_directory = os.getcwd()
    files_directory = os.path.join(current_directory, directory_name)
    if not os.path.exists(files_directory):
        os.makedirs(files_directory)
    return files_directory

def save_czi_files(images): # TODO: saving data to folder
    czi_files_directory = create_directory("czi_files")

    for img in images:
        open(os.path.join(czi_files_directory, img['filename']), 'wb').write(img['content'])


def get_czi_file_names(images):
    czi_file_names = []

    for img in images:
        czi_file_names.append(img['filename'])

    return czi_file_names

def czi_to_jpg(czi_files, czi_file_names):
    images_directory = create_directory("images")

    index = 0
    while index < len(czi_files):
        fn_path = Path(os.path.join(os.getcwd(), "czi_files", czi_file_names[index]))
        fn_str = str(fn_path)
        if not fn_path.exists():
            break
        print(f"filename: {fn_path} {fn_path.exists()}")

        anim = scaffan.image.AnnotatedImage(path=fn_str)

        view = anim.get_full_view(
            pixelsize_mm=[0.0003, 0.0003]
        )  # wanted pixelsize in mm in view
        img = view.get_raster_image()
        skimage.io.imsave(os.path.join(images_directory, str(index).zfill(4) + ".jpg"), img)
        index += 1


def create_COCO_json():
    # Directory of the image dataset
    dataset_directory = Path(os.path.join(os.getcwd(), "images"))

    # Directory of the .czi files
    czi_files_directory = Path(os.path.join(os.getcwd(), "czi_files"))  # path to .czi files directory

    data = {}

    """
    Info
    """
    version = "1.0"
    description = "COCO dataset for scaffan"
    contributor = "Jan Burian" # TODO: chnage contributor (user info)

    info_dictionary = COCO_json.get_info_dictionary(version, description, contributor)
    data.update({"info": info_dictionary})

    """
    Images
    """
    list_image_dictionaries = COCO_json.get_image_properties(dataset_directory)
    data.update({"images": list_image_dictionaries})

    """
    Categories
    """
    list_category_dictionaries = COCO_json.get_category_properties(
        dataset_directory, "categories.txt"
    )  # directory and .txt file
    data.update({"categories": list_category_dictionaries})

    """
    Annotations
    """
    annotation_name = "annotation" # TODO: change of annotation names
    pixelsize_mm = [0.0003, 0.0003]
    list_annotation_dictionaries = COCO_json.get_annotations_properties(
        czi_files_directory, annotation_name, pixelsize_mm
    )
    data.update({"annotations": list_annotation_dictionaries})

    return data


def copy_images():
    source_dir = os.path.join(os.getcwd(), "images")
    destination_dir = os.path.join(os.getcwd(), "COCO_dataset", "images")
    shutil.copytree(source_dir, destination_dir)


def create_COCO_dataset():
    json_COCO = create_COCO_json()

    COCO_directory = create_directory("COCO_dataset")

    name_json = "trainval"

    # Creating .json file
    with open(os.path.join(COCO_directory, name_json), "w", encoding="utf-8") as f:
        json.dump(json_COCO, f, ensure_ascii=False, indent=4)
        f.close()

    copy_images()


def get_categories():
    categories = input("Please, write category/categories separated by commas: ", required=True)
    return categories


def create_txt_categories_file(list_categories):
    txt_file_path = os.path.join(os.getcwd(), "images", "categories.txt")
    with open(txt_file_path, 'w') as f:
        for category in list_categories:
            f.write(category)
            f.write("\n")
    f.close()


if __name__=="__main__":
    user_info = get_user_info()
    data = upload_data_page()

    czi_files = data['imgs']
    operation = data['operation']

    save_czi_files(czi_files)
    czi_files_names = get_czi_file_names(czi_files)

    czi_to_jpg(czi_files, czi_files_names)

    if (operation == 'Predict'):
        model_option = select("Choose one of pretrained models or upload your own", ['model_1', 'model_2', 'model_3', 'upload_own'], required=True),

        if (model_option[0] == 'upload_own'):
           own_model = file_upload("Upload your own model:", accept=".pth")
           open(own_model['filename'], 'wb').write(own_model['content'])

    elif (operation == 'Train'):
        categories = get_categories().split(", ")
        create_txt_categories_file(categories)

        create_COCO_dataset()
        pass



    print()





