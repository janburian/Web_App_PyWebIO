import glob
import math
import os
import sys
import zipfile
import shutil

import pywebio.session
import skimage.io

from pywebio.input import *
from pywebio.output import *
import re
from PIL import Image

from pywebio.pin import *
from pywebio.utils import random_str

from pathlib import Path
import pathlib

import COCO_json
import detectron2_testovaci

path_to_script = Path("~/GitHub/scaffan").expanduser()
print(os.path.exists(path_to_script))
sys.path.insert(0, str(path_to_script))
import scaffan.image
import json


def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if (re.fullmatch(regex, email)):
        return
    return "Bad format of email."


def get_user_info():
    put_html("""
            <h1>Description</h1>
            <p>The main goal of this computer vision application is to detect cell nuclei in microscopic histological images.
            It is based on Detectron2 framework, so for the detection it uses deep neural network. 
            User also can train his own model through this application, too. </p>
            
            <p>The application was written in Python and it is powered by PyWebIO module.</p>
            
            <h1>How it works (prediction)</h1> """)

    img = Image.open("schema2.png", 'r')
    #img = Image.open("schema_train_pred.png", 'r')
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

def delete_model_page():
    data = input_group('List of models to delete (optional)', [
        checkbox("Available models", name='deleted_models', options=get_specific_files("models", ".pth")),
        actions('', [
            {'label': 'Delete', 'value': 'delete', 'disabled': False, 'color': 'danger'},
            {'label': 'Reset', 'type': 'reset', 'color': 'warning'},
            {'label': 'Continue', 'value': 'continue', 'color': 'primary'},
        ], name='action'),
    ])

    if data['action'] == "delete":
        models_to_delete = data['deleted_models']
        delete_models(models_to_delete)
        delete_model_page()

def create_directory(directory_name):
    files_directory = os.path.join(Path(__file__).parent, directory_name)
    if not os.path.exists(files_directory):
        os.makedirs(files_directory, exist_ok=True)
    return files_directory


def delete_models(models_to_delete: list):
    path_models = os.path.join(Path(__file__).parent, "models")
    for model in models_to_delete:
        model_path = os.path.join(path_models, model)
        if os.path.exists(model_path):
            os.remove(model_path)


def save_czi_files(czi_files, directory_name: str):
    czi_files_directory = create_directory(directory_name)

    for file in czi_files:
        open(os.path.join(czi_files_directory, file['filename']), 'wb').write(file['content'])


def get_czi_file_names(images):
    czi_file_names = []

    for img in images:
        czi_file_names.append(img['filename'])

    return czi_file_names


def czi_to_jpg(czi_files, czi_file_names, imgs_directory_name: str, czi_directory_name: str):
    images_directory = create_directory(imgs_directory_name)

    index = 0
    while index < len(czi_files):
        fn_path = Path(os.path.join(Path(__file__).parent, czi_directory_name, czi_file_names[index]))
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


def create_COCO_json(czi_files_names, images_names: list, user_info, images_directory: str, czi_files_directory: str):
    # Directory of the image dataset
    dataset_directory = images_directory

    # Directory of the .czi files
    czi_files_directory = czi_files_directory  # path to .czi files directory

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
    list_image_dictionaries = COCO_json.get_image_properties(dataset_directory, images_names)
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


def copy_images(images_names: list, COCO_dir_name: str, source_dir: str):
    os.mkdir(os.path.join(Path(__file__).parent, COCO_dir_name, "images"))
    destination_dir = os.path.join(Path(__file__).parent, COCO_dir_name, "images")

    index = 0
    while index < len(images_names):
        image_name = images_names[index]
        full_file_name = os.path.join(source_dir, image_name)
        if os.path.exists(full_file_name):
            shutil.copy(full_file_name, destination_dir)
            index += 1


def create_COCO_dataset(czi_files_names, images_names: list, user_info, COCO_dir_name: str, images_directory: str, czi_files_directory: str):
    name_json = "trainval.json"
    json_COCO = create_COCO_json(czi_files_names, images_names, user_info, images_directory, czi_files_directory)

    COCO_directory = create_directory(COCO_dir_name)

    # Creating .json file
    with open(os.path.join(COCO_directory, name_json), "w", encoding="utf-8") as f:
        json.dump(json_COCO, f, ensure_ascii=False, indent=4)
        f.close()

    copy_images(images_names, COCO_dir_name, images_directory)


def get_categories():
    categories = input("Please, write category/categories separated by commas: ", required=True)
    return categories


def create_save_txt_categories_file(list_categories, directory_name: str):
    txt_file_path = os.path.join(Path(__file__).parent, directory_name, "categories.txt")
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


def get_specific_files(directory_name: str, ending: str):
    files = glob.glob(os.path.join(Path(__file__).parent, directory_name, "*" + ending))
    files_list = []
    for file in files:
        files_list.append(os.path.basename(file))

    return files_list


def choose_model(available_models):
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

def merge_zip_file(zips:list):
    """
    Open the first zip file as append and then read all
    subsequent zip files and append to the first one
    """
    with zipfile.ZipFile(zips[0], 'a') as z1:
        for fname in zips[1:]:
            zf = zipfile.ZipFile(fname, 'r')
            for n in zf.namelist():
                z1.writestr(n, zf.open(n).read())

def delete_zip_files():
    dir_name = os.path.join(Path(__file__).parent)
    files = os.listdir(dir_name)

    for file in files:
        if file.endswith(".zip"):
            os.remove(os.path.join(dir_name, file))


def visualize_predictions():
    put_text("Predicted data visualization").style('font-size: 20px')
    processed_images_list = os.listdir(os.path.join(Path(__file__).parent, "processed", "vis_predictions"))
    for image in processed_images_list:
        put_table([
            [put_image(Image.open(os.path.join(Path(__file__).parent, "processed", "vis_predictions", image), 'r'),
                       title=image)],
        ])

def visualize_annotated_data():
    put_text("Annotated training data visualization").style('font-size: 20px')
    processed_images_list = os.listdir(os.path.join(Path(__file__).parent, "processed", "vis_train"))
    for image in processed_images_list:
        put_table([
            [put_image(Image.open(os.path.join(Path(__file__).parent, "processed", "vis_train", image), 'r'),
                       title=image)],
        ])


def put_downloadable_files():
    content_output = open('output.zip', 'rb').read()
    put_file('output.zip', content_output, 'Download trained model/s.')
    content_data = open('data.zip', 'rb').read()
    put_file('data.zip', content_data, 'Download image data with annotations.')
    content_COCO_compl = open('COCO_complete.zip', 'rb').read()
    put_file('COCO_complete.zip', content_COCO_compl, 'Download complete custom COCO dataset.')
    content_COCO_tra = open('COCO_training.zip', 'rb').read()
    put_file('COCO_training.zip', content_COCO_tra, 'Download training COCO dataset.')
    content_COCO_val = open('COCO_test.zip', 'rb').read()
    put_file('COCO_test.zip', content_COCO_val, 'Download validation COCO dataset.')


def create_training_test_datasets():
    number_train = math.floor((len(czi_files) / 100) * 80)
    number_validate = len(czi_files) - number_train
    czi_names_train = []

    for i in range(number_train):
        czi_names_train.append(czi_files_names[i])

    czi_names_validate = list(set(czi_files_names).symmetric_difference(set(czi_names_train)))
    images_list = os.listdir(os.path.join(Path(__file__).parent, "images"))
    images_names_train = []
    for i in range(len(czi_names_train)):
        images_names_train.append(images_list[i])

    images_names_validate = list(set(images_names_train).symmetric_difference(set(images_list)))
    images_names_validate.remove("categories.txt")

    images_directory = os.path.join(Path(__file__).parent, "images")
    czi_files_directory = os.path.join(Path(__file__).parent, "czi_files")

    create_COCO_dataset(czi_names_train, images_names_train, user_info, "COCO_train", images_directory, czi_files_directory)
    create_COCO_dataset(czi_names_validate, images_names_validate, user_info, "COCO_test", images_directory, czi_files_directory)

def upload_data_test_dataset_page():
    data = input_group("User can upload annotated files in .czi format to create the test/validation dataset, otherwise the test dataset will be created automatically: ", [
        file_upload("Upload data", accept=".czi", multiple=True, required=False, name="czi"),
        actions('', [
            #{'label': 'Submit', 'type': 'submit', 'value': 'submit'},
            {'label': 'Continue', 'value': 'continue', 'color': 'primary'},
        ], name='action'),
    ])

    return data

def copy_file(filename: str, source_dir: str, destination_dir: str, new_filename: str):
    full_filename = os.path.join(source_dir, filename)
    full_filename_new = os.path.join(source_dir, new_filename)
    if os.path.exists(full_filename):
        os.rename(full_filename, full_filename_new)
        shutil.copy(full_filename_new, destination_dir)


def popup_input(pins, title='Please write name of the trained model. After that the trained model will be added to available models directory.'):
    """Show a form in popup window.

    :param list pins: pin output list.
    :param str title: model title.
    :return: return the form value as dict, return None when user cancel the form.
    """
    if not isinstance(pins, list):
        pins = [pins]

    pin_names = [
        p.spec['input']['name']
        for p in pins
    ]
    action_name = 'action_' + random_str(10)
    pins.append(put_actions(action_name, buttons=[
        {'label': 'Submit', 'value': 'submit', 'disabled': False},
    ]))
    popup(title=title, content=pins, closable=False)

    change_info = pin_wait_change(action_name)
    result = None
    if change_info['name'] == action_name and change_info['value']:
        result = {name: pin[name] for name in pin_names}

    model_name = result['model_name']
    action = change_info['value']

    if action == 'submit' and model_name != "":
       close_popup()
       return model_name

    else:
        popup_input([put_input(name='model_name')])


def get_model_name():
    pins = [put_input(name='model_name')]
    model_name = popup_input(pins)

    return model_name


if __name__ == "__main__":
    delete_content_folder(os.path.join(Path(__file__).parent, "czi_files"))
    delete_content_folder(os.path.join(Path(__file__).parent, "images"))
    delete_content_folder(os.path.join(Path(__file__).parent, "processed"))
    delete_content_folder(os.path.join(Path(__file__).parent, "output"))
    delete_content_folder(os.path.join(Path(__file__).parent, "COCO_train"))
    delete_content_folder(os.path.join(Path(__file__).parent, "COCO_test"))
    delete_content_folder(os.path.join(Path(__file__).parent, "COCO_complete"))
    delete_content_folder(os.path.join(Path(__file__).parent, "masks_prediction"))
    delete_content_folder(os.path.join(Path(__file__).parent, "images_test"))
    delete_content_folder(os.path.join(Path(__file__).parent, "czi_files_test"))

    delete_zip_files()

    user_info = get_user_info()
    data = upload_data_page()

    czi_files = data['czi']
    operation = data['operation']

    save_czi_files(czi_files, "czi_files")
    czi_files_names = get_czi_file_names(czi_files)

    czi_to_jpg(czi_files, czi_files_names, "images", "czi_files")

    if (operation == 'Predict'):
        delete_model_page()
        available_models = get_specific_files("models", ".pth")
        available_models.append('upload_own')
        model_option = choose_model(available_models)

        model_name = model_option

        if model_option == "upload_own":
            model_name = save_own_model(model_option)

        with put_loading("border", "primary"):
            detectron2_testovaci.predict(os.path.join(Path(__file__).parent / "images"), os.path.join(Path(__file__).parent), model_name)

        visualize_predictions()

        processed_dir_path = os.path.join(Path(__file__).parent, "processed")
        create_zip_directory(processed_dir_path, "results.zip")
        content = open('results.zip', 'rb').read()
        put_file('results.zip', content, 'Download results.')
        pywebio.session.hold()

    elif (operation == 'Train'):
        #available_models = get_available_models()
        #available_models.append('upload_own')
        #model_option = choose_model()

        #model_name = model_option

        #if model_option == "upload_own":
            #model_name = save_own_model(model_option)

        data_test = upload_data_test_dataset_page() # TODO: test dataset
        categories = get_categories().split(", ")
        czi_files_test = data_test['czi']

        if len(czi_files_test) > 0: # user wants to use his own test dataset
            save_czi_files(czi_files_test, "czi_files_test")
            czi_files_names_test = get_czi_file_names(czi_files_test)
            czi_to_jpg(czi_files_test, czi_files_names_test, "images_test", "czi_files_test")
            create_save_txt_categories_file(categories, "images_test")
            create_save_txt_categories_file(categories, "images")

            images_names_test = get_specific_files("images_test", ".jpg")
            images_names_train = get_specific_files("images", ".jpg")

            images_test_directory = os.path.join(Path(__file__).parent, "images_test")
            czi_files_test_directory = os.path.join(Path(__file__).parent, "czi_files_test")

            images_train_directory = os.path.join(Path(__file__).parent, "images")
            czi_files_train_directory = os.path.join(Path(__file__).parent, "czi_files")

            create_COCO_dataset(czi_files_names_test, images_names_test, user_info, "COCO_test", images_test_directory, czi_files_test_directory)
            create_COCO_dataset(czi_files_names, images_names_train, user_info, "COCO_train", images_train_directory, czi_files_train_directory)

        else: # dataset is created automatically
            create_save_txt_categories_file(categories, "images")
            if len(czi_files) > 1:
                create_training_test_datasets()

            images_names_all = os.listdir(os.path.join(Path(__file__).parent, "images"))
            images_names_all.remove("categories.txt")

            images_directory = os.path.join(Path(__file__).parent, "images")
            czi_files_directory = os.path.join(Path(__file__).parent, "czi_files")
            create_COCO_dataset(czi_files_names, images_names_all, user_info, "COCO_complete", images_directory, czi_files_directory) # complete COCO (training + validation data)

        #parameters = define_detectron2_parameters()

        processed = create_directory("processed")

        COCO_train_path = os.path.join(Path(__file__).parent, "COCO_train")
        COCO_validation_path = os.path.join(Path(__file__).parent, "COCO_test")

        if len(czi_files) > 1:
            cells_metadata, dataset_dicts = detectron2_testovaci.register_coco_instances(COCO_train_path, COCO_validation_path)
        else:
            cells_metadata, dataset_dicts = detectron2_testovaci.register_coco_instances(os.path.join(Path(__file__).parent, "COCO_complete"),
                                                                                         COCO_validation_path)
        with put_loading("border", "primary"):
            detectron2_testovaci.check_annotated_data(os.path.join(Path(__file__).parent, "processed"), cells_metadata, dataset_dicts)

        visualize_annotated_data()

        with put_loading("border", "primary"):
            detectron2_testovaci.train()

        trained_model_name = get_model_name()
        copy_file("model_final.pth", os.path.join(Path(__file__).parent, "output"), os.path.join(Path(__file__).parent, "models"), trained_model_name + ".pth")

        output_path = os.path.join(Path(__file__).parent, "output")
        processed_dir_path = os.path.join(Path(__file__).parent, "processed")
        COCO_path_complete = os.path.join(Path(__file__).parent, "COCO_complete")
        COCO_path_train = os.path.join(Path(__file__).parent, "COCO_train")
        COCO_path_validation = os.path.join(Path(__file__).parent, "COCO_test")

        create_zip_directory(output_path, "output.zip")
        create_zip_directory(processed_dir_path, "data.zip")
        create_zip_directory(COCO_path_complete, "COCO_complete.zip")
        create_zip_directory(COCO_path_train, "COCO_training.zip")
        create_zip_directory(COCO_path_validation, "COCO_test.zip")

        put_downloadable_files()

        pywebio.session.hold()

