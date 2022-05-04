import pywebio.session
from pywebio.input import *
import os
from pathlib import Path
import glob


def get_available_models():
    models = glob.glob(os.path.join(Path(__file__).parent, "models", "*.pth"))
    models_list = []
    for model in models:
        models_list.append(os.path.basename(model))

    return models_list


def show_delete_models_form(models: list):
    info = input_group('Models', [
        checkbox("Available models", name='deleted_models', options=models),
        actions('', [
            {'label': 'Delete', 'value': 'delete', 'disabled': False},
            {'label': 'Reset', 'type': 'reset', 'color': 'warning'},
            {'label': 'Cancel', 'type': 'cancel', 'color': 'danger'},
            {'label': 'Upload', 'value': 'upload', 'disabled': False},
        ], name='action'),
    ])

    return info


def delete_models(models_to_delete: list):
    path_models = os.path.join(Path(__file__).parent, "models")
    for model in models_to_delete:
        model_path = os.path.join(path_models, model)
        if os.path.exists(model_path):
            os.remove(model_path)


def choose_action(action: str, models_to_delete: list):
    if action == "delete":
        delete_models(models_to_delete)
    if action == "cancel":
        pass

models = get_available_models()
info_models_delete = show_delete_models_form(models)
while (len(get_available_models()) > 0):
    if info_models_delete == None:
        choose_action("cancel", [])
    action = info_models_delete["action"]
    models_to_delete = info_models_delete["deleted_models"]
    choose_action(action, models_to_delete)
    info_models_delete = show_delete_models_form(get_available_models())

pywebio.session.hold()
