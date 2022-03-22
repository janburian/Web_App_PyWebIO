from pywebio.input import *
from pywebio.output import *
import re

def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if (re.fullmatch(regex, email)):
        return
    return "Bad email."

def get_user_info():
    info = input_group("User info",[
      input('Input your name', name='name', required=True),
      input('Input your email', name='email', type=TEXT, required=True, validate=check_email)
    ])
    put_text(info['name'], info['email'])
    return info

def upload_files():
    imgs = file_upload("Upload .czi files:", accept=".czi", multiple=True)
    for img in imgs:
        put_image(img['content']) #TODO: save pictures

def choose_operation():
    # predict, train
    pass

def choose_model():
    pass


if __name__=="__main__":
    user_info = get_user_info()
    upload_files()

    print()





