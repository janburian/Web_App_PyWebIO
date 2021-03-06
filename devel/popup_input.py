import pywebio.session
from pywebio.input import *
from pywebio.output import *
import os
from pywebio.pin import *
from pywebio.utils import random_str
from pathlib import Path
import glob

def get_available_models():
    models = glob.glob(os.path.join(Path(__file__).parent, "models", "*.pth"))
    models_list = []
    for model in models:
        models_list.append(os.path.basename(model))

    return models_list

def delete_models(models_to_delete: list):
    path_models = os.path.join(Path(__file__).parent, "models")
    for model in models_to_delete:
        model_path = os.path.join(path_models, model)
        if os.path.exists(model_path):
            os.remove(model_path)


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
    choose_action(action, model_name)


def choose_action(action, model_name):
    if action == 'submit' and model_name != "":
       close_popup()

    else:
        popup_input([put_input(name='model_name')])


#pins = [put_checkbox(name='chosen_models', options=get_available_models())]
pins = [put_input(name='model_name')]
popup_input(pins)

pywebio.session.hold()