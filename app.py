import os
import sys
import skimage.io

from pywebio.input import *
from pywebio.output import *
import re

from pathlib import Path


path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image



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

def save_czi_files(images): # TODO: saving data to folder
    for img in images:
        open(img['filename'], 'wb').write(img['content'])

def get_czi_file_names(images):
    czi_file_names = []

    for img in images:
        czi_file_names.append(img['filename'])

    return czi_file_names

def czi_to_jpg(czi_files, czi_file_names):
    path_images = Path(
        r"C:\Users\janbu\Desktop"
    )  # path to directory, where the images will be saved

    index = 0
    while index < len(czi_files):
        fn_str = (czi_file_names[index])
        fn_path = Path(fn_str)
        if not fn_path.exists():
            break
        print(f"filename: {fn_path} {fn_path.exists()}")

        anim = scaffan.image.AnnotatedImage(path=fn_str)

        view = anim.get_full_view(
            pixelsize_mm=[0.0003, 0.0003]
        )  # wanted pixelsize in mm in view
        img = view.get_raster_image()
        os.makedirs(path_images, exist_ok=True)
        skimage.io.imsave(os.path.join(path_images, str(index).zfill(4) + ".jpg"), img)
        index += 1


if __name__=="__main__":
    user_info = get_user_info()
    data = upload_data_page()

    czi_files = data['imgs']
    operation = data['operation']

    model_option = ""

    save_czi_files(czi_files)
    czi_files_names = get_czi_file_names(czi_files)

    if (operation == 'Predict'):
        model_option = select("Choose one of pretrained models or upload your own", ['model_1', 'model_2', 'model_3', 'upload_own'], required=True),

        if (model_option[0] == 'upload_own'):
           own_model = file_upload("Upload your own model:", accept=".pth")
           open(own_model['filename'], 'wb').write(own_model['content'])

    elif (operation == 'Train'):
        pass

    czi_to_jpg(czi_files, czi_files_names)

    print()





