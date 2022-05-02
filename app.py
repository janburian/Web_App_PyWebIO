import glob
import os
import random
import sys
import zipfile
import shutil

import skimage.io

from pywebio.input import *
from pywebio.output import *
import re
import pywebio
from PIL import Image

from pathlib import Path

import COCO_json
import detectron2_testovaci

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image
import imma.image

import json
import distutils
from distutils import dir_util


def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if (re.fullmatch(regex, email)):
        return
    return "Bad format of email."


def get_user_info():
    put_html("""<!DOCTYPE html>
                    <html>
                    <head>
                    <title>Page Title</title>
                    </head>
                    <body>
                    
                    <h1>Description</h1>
                    <p>The main goal of this computer vision application is to detect cell nuclei in microscopic histological images.
                    It is based on Detectron2 framework, so for the detection it uses deep neural network. 
                    User also can train his own model through this application, too. </p>
                    <p>The application was written in Python and it is powered by PyWebIO module.</p>
                    
                    <h1>How it works</h1>
                    </body>
                    </html>""")

    img = Image.open("schema2.png", 'r')
    put_image(img, width='903px', height='160px')

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
    files_directory = os.path.join(Path(__file__).parent, directory_name)
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
        fn_path = Path(os.path.join(Path(__file__).parent, "czi_files", czi_file_names[index]))
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
    dataset_directory = Path(os.path.join(Path(__file__).parent, "images"))

    # Directory of the .czi files
    czi_files_directory = Path(os.path.join(Path(__file__).parent, "czi_files"))  # path to .czi files directory

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
        popup('Warning', 'No annotations in .czi files.')

    data.update({"annotations": list_annotation_dictionaries})

    return data


def copy_images():
    source_dir = os.path.join(Path(__file__).parent, "images")
    destination_dir = os.path.join(Path(__file__).parent, "COCO_dataset", "images")
    distutils.dir_util.copy_tree(source_dir, destination_dir)


def create_COCO_dataset(czi_files_names, user_info):
    name_json = "trainval.json"
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
    txt_file_path = os.path.join(Path(__file__).parent, "images", "categories.txt")
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
    models = glob.glob(os.path.join(Path(__file__).parent, "models", "*.pth"))
    models_list = []
    for model in models:
        models_list.append(os.path.basename(model))

    return models_list


def choose_model():
    global model_option
    model_option = select("Choose one of pretrained models or upload your own",
                          available_models, required=True)
    return model_option


def save_own_model(model_option):
    if (model_option == 'upload_own'):
        own_model = file_upload("Upload your own model:", accept=".pth")
        open(own_model['filename'], 'wb').write(own_model['content'])

    return own_model['filename']

def create_zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, mode='w') as zipf:
        len_dir_path = len(folder_path)
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len_dir_path:])
    zipf.close()

def download_data():
    processed_dir_path = os.path.join(Path(__file__).parent, "processed")
    create_zip_directory(processed_dir_path, "data.zip")
    #f = open("data.zip", 'rb')
    #content = f.read()
    #f.close()
    put_file(os.path.join(Path(__file__).parent, "data.zip"))


def delete_content_folder(path_folder: str):
    if os.path.exists(path_folder):
        for filename in os.listdir(path_folder):
            file_path = os.path.join(path_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":
    delete_content_folder(os.path.join(Path(__file__).parent, "czi_files"))
    delete_content_folder(os.path.join(Path(__file__).parent, "images"))
    delete_content_folder(os.path.join(Path(__file__).parent, "processed"))

    user_info = get_user_info()
    data = upload_data_page()

    czi_files = data['czi']
    operation = data['operation']

    save_czi_files(czi_files)
    czi_files_names = get_czi_file_names(czi_files)

    czi_to_jpg(czi_files, czi_files_names)

    ''' Test
    anim = scaffan.image.AnnotatedImage(path=os.path.join(Path(__file__).parent, "czi_files", "test.czi"))
    view = anim.get_full_view(
        pixelsize_mm=[0.0003, 0.0003]
    )  # wanted pixelsize in mm in view
    annotations = view.annotations
    '''

    if (operation == 'Predict'):
        available_models = get_available_models()
        available_models.append('upload_own')
        model_option = choose_model()

        model_name = model_option

        if model_option == "upload_own":
            model_name = save_own_model(model_option)

        with put_loading("border", "primary"):
            detectron2_testovaci.predict(os.path.join(Path(__file__).parent / "images"), os.path.join(Path(__file__).parent), model_name)

        put_text("Predicted data visualization").style('font-size: 20px')
        put_table([
            [put_image(Image.open(os.path.join(Path(__file__).parent, "processed", "vis_predictions", "pic_pred_0000.jpg"), 'r')),
             put_image(Image.open(os.path.join(Path(__file__).parent, "processed", "vis_predictions", "pic_pred_0001.jpg"), 'r'))],
        ])

        # TODO: Visualization of predicted data
        #download_data()
        #test = choose_model()

        #pywebio.session.hold()

    elif (operation == 'Train'):
        available_models = get_available_models()
        available_models.append('upload_own')
        model_option = choose_model()

        model_name = model_option

        if model_option == "upload_own":
            model_name = save_own_model(model_option)

        categories = get_categories().split(", ")
        create_txt_categories_file(categories)

        create_COCO_dataset(czi_files_names, user_info)

        parameters = define_detectron2_parameters()

        processed = create_directory("processed")
        cells_metadata, dataset_dicts = detectron2_testovaci.register_coco_instances(os.path.join(Path(__file__).parent, "COCO_dataset"))
        with put_loading("border", "primary"):
            detectron2_testovaci.check_annotated_data(os.path.join(Path(__file__).parent, "processed"), cells_metadata, dataset_dicts)

        put_text("Annotated data visualization").style('font-size: 20px')

        put_table([
            [put_image(Image.open(os.path.join(Path(__file__).parent, "processed", "vis_train", "0000.jpg"), 'r')),
             put_image(Image.open(os.path.join(Path(__file__).parent, "processed", "vis_train", "0001.jpg"), 'r'))],
        ])

        with put_loading("border", "primary"):
            detectron2_testovaci.train(model_name)

        #download_data()
        print()

