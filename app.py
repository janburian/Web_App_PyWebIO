import os

from pywebio.input import *
from pywebio.output import *
import re

import czi_to_jpg


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
    for i in range(len(images)):
        img = images[i]
        open(img['filename'], 'wb').write(img['content'])

def processing_data():
    pass


if __name__=="__main__":
    user_info = get_user_info()
    data = upload_data_page()

    czi_files = data['imgs']
    operation = data['operation']

    model_option = ""

    save_czi_files(czi_files)

    if (operation == 'Predict'):
        model_option = select("Choose model or upload your own", ['model_1', 'model_2', 'model_3', 'upload_own']),

        if (model_option[0] == 'upload_own'):
           own_model = file_upload("Upload your own model:", accept=".pth")
           open(own_model['filename'], 'wb').write(own_model['content'])



    processing_data()

    print()





