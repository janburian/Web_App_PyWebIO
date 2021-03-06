import glob
import os
import sys

import skimage.io

from pywebio.input import *
from pywebio.output import *
import re

from pathlib import Path

from src import COCO_json

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image

import json
import distutils
from distutils import dir_util


def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if (re.fullmatch(regex, email)):
        return
    return "Bad format of email."


def get_user_info():
    info = input_group("User info", [
        input('Enter your username', name='username', required=True),
        input('Enter your email', name='email', type=TEXT, required=True, validate=check_email)
    ])
    put_text(info['username'], info['email'])

    return info


def upload_data_page():
    data = input_group("Main page", [
        file_upload("Upload .czi files:", accept=".czi", multiple=True, required=True, name="czi"),
        radio("Choose one option", options=['Predict', 'Train'], required=True, name="operation"),
        textarea('Description', rows=3, name="description"),
    ])

    return data


def create_directory(directory_name):
    current_directory = os.getcwd()
    files_directory = os.path.join(current_directory, directory_name)
    if not os.path.exists(files_directory):
        os.makedirs(files_directory, exist_ok=True)
    return files_directory


def save_czi_files(czi_files):
    czi_files_directory = create_directory("czi_files")

    for file in czi_files:
        open(os.path.join(czi_files_directory, file['filename']), 'wb').write(file['content'])


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


def create_COCO_json(czi_files_names, user_info):
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
    contributor = user_info['username']

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
    pixelsize_mm = [0.0003, 0.0003]
    list_annotation_dictionaries = COCO_json.get_annotations_properties(
        czi_files_directory, czi_files_names, pixelsize_mm
    )
    if (len(list_annotation_dictionaries) == 0):
        popup('Warning', 'No annotations in czi files.')

    data.update({"annotations": list_annotation_dictionaries})

    return data


def copy_images():
    source_dir = os.path.join(os.getcwd(), "images")
    destination_dir = os.path.join(os.getcwd(), "COCO_dataset", "images")
    distutils.dir_util.copy_tree(source_dir, destination_dir)


def create_COCO_dataset(czi_files_names, user_info):
    name_json = "trainval"
    json_COCO = create_COCO_json(czi_files_names, user_info)

    COCO_directory = create_directory("COCO_dataset")

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


def define_detectron2_parameters():
    parameters = input_group("Detectron2 parameters: ", [
        input('SOLVER.IMS_PER_BATCH', name='ims_per_batch', placeholder="2", type=NUMBER),
        input('SOLVER.BASE_LR', name='base_lr', placeholder="0.000002", type=NUMBER),
        input('SOLVER.MAX_ITER', name='max_iter', placeholder="120000", type=NUMBER),
        input('MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE', placeholder="128", name='batch_size', type=NUMBER),
    ])

    return parameters


def get_available_models():
    models = glob.glob(os.path.join(os.getcwd(), "models", "*.pth"))
    models_list = []
    for model in models:
        models_list.append(os.path.basename(model))

    return models_list


if __name__ == "__main__":
    user_info = get_user_info()
    data = upload_data_page()

    czi_files = data['czi']
    operation = data['operation']

    save_czi_files(czi_files)
    czi_files_names = get_czi_file_names(czi_files)

    czi_to_jpg(czi_files, czi_files_names)

    ''' Test
    anim = scaffan.image.AnnotatedImage(path=os.path.join(os.getcwd(), "czi_files", "test.czi"))
    view = anim.get_full_view(
        pixelsize_mm=[0.0003, 0.0003]
    )  # wanted pixelsize in mm in view
    annotations = view.annotations
    '''

    if (operation == 'Predict'):
        available_models = get_available_models()

        model_pred = input_group("Choosing model for prediction: ", [
            select("Choose one of pretrained models or upload your own", available_models, placeholder=" ", name='chosen_model'),
            file_upload("Upload your own model:", accept=".pth", name='own_model'),
        ])

        if (model_pred['own_model'] != None):
            own_model_name = input("Name of uploaded model: ", type=TEXT)
            own_model = model_pred['own_model']
            open(os.rename(own_model['filename'], own_model_name + ".pth"), 'wb').write(own_model['content'])

    elif (operation == 'Train'):
        categories = get_categories().split(", ")
        create_txt_categories_file(categories)

        create_COCO_dataset(czi_files_names, user_info)

        parameters = define_detectron2_parameters()

    print()
