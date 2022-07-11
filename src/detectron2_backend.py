# **Usage of Detectron2**
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from matplotlib import pyplot as plt

# import some common libraries
import numpy as np
import cv2
import glob
import os
from pathlib import Path
import json

import COCO_json

setup_logger()

def create_directory(directory_name):
    files_directory = os.path.join(Path(__file__).parent, directory_name)
    if not os.path.exists(files_directory):
        os.makedirs(files_directory)
    return files_directory


def register_coco_instances(COCO_train_path, COCO_test_path):
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("cells_training", {}, os.path.join(COCO_train_path, "trainval.json"), os.path.join(COCO_train_path, "images"))
    cells_metadata = MetadataCatalog.get("cells_training")
    dataset_dicts = DatasetCatalog.get("cells_training")

    if os.path.exists(COCO_test_path):
        register_coco_instances("cells_test", {}, os.path.join(COCO_test_path, "trainval.json"), os.path.join(COCO_test_path, "images"))

    return cells_metadata, dataset_dicts


def check_annotated_data(output_data_path, cells_metadata, dataset_dicts):
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=cells_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        Path(os.path.join(output_data_path, "vis_train")).mkdir(parents=True, exist_ok=True)
        img_name_final = os.path.basename(d["file_name"])
        if not cv2.imwrite(os.path.join(output_data_path, "vis_train", img_name_final), vis.get_image()[:, :, ::-1]):
            raise Exception("Could not write image: " + img_name_final)


def train():
    from detectron2.engine import DefaultTrainer

    # Parameters
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(
        r"C:\Users\janbu\miniconda3\envs\scaffan_2\Lib\site-packages\detectron2\model_zoo\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("cells_training",)
    if len(os.listdir(os.path.join(Path(__file__).parent, "czi_files"))) > 1:
        cfg.DATASETS.TEST = ("cells_test",)
    else:
        cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    #cfg.MODEL.WEIGHTS = os.path.join(Path(__file__).parent, "models", model_name)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00005
    cfg.SOLVER.MAX_ITER = 5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (cells nuclei)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(os.path.abspath(cfg.OUTPUT_DIR))
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



def predict(input_data_dir, output_data_dir, model_name: str):
    cfg = get_cfg()
    config_name = os.path.splitext(model_name)[0]
    pretrained_models_original = ["mask_rcnn_R_101_C4_3x.pth", "mask_rcnn_R_101_DC5_3x.pth", "mask_rcnn_R_101_FPN_3x.pth", "mask_rcnn_R_50_C4_3x.pth", "mask_rcnn_R_50_DC5_3x.pth", "mask_rcnn_R_50_FPN_3x.pth"]
    if model_name in pretrained_models_original:
        cfg.merge_from_file(os.path.join( r"C:\Users\janbu\miniconda3\envs\scaffan_2\Lib\site-packages\detectron2\model_zoo\configs\COCO-InstanceSegmentation", config_name + ".yaml"))
    else:
        cfg.merge_from_file(r"C:\Users\janbu\miniconda3\envs\scaffan_2\Lib\site-packages\detectron2\model_zoo\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml") # one architecture
    cfg.MODEL.WEIGHTS = os.path.join(Path(__file__).parent, "models", model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (cells nuclei)
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    picture_predictions = glob.glob(str(input_data_dir) + "/*.jpg")
    number_pictures_predictions = len(picture_predictions)

    print("Number of pictures for prediction: " + str(number_pictures_predictions))
    index = 0
    outputs_list = []
    img_names_list = []
    for d in range(number_pictures_predictions):
        index_str = str(index)
        # print("TEST: " + str(input_data_dir_predict) + "/" + index_str.zfill(4) + ".jpg")
        print(os.path.join(input_data_dir, index_str.zfill(4), ".jpg"))
        im = cv2.imread(os.path.join(input_data_dir, index_str.zfill(4) + ".jpg"))
        outputs = predictor(im)
        outputs_list.append(outputs)
        v = Visualizer(im[:, :, ::-1],
                       #metadata=cells_metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        Path(os.path.join(output_data_dir, "processed", "vis_predictions")).mkdir(parents=True, exist_ok=True)
        img_name_final = "pic_pred_" + index_str.zfill(4) + ".jpg"
        img_names_list.append(img_name_final)
        index += 1
        if not cv2.imwrite(os.path.join(output_data_dir, "processed", "vis_predictions", img_name_final), v.get_image()[:, :, ::-1]):
            raise Exception("Could not write image: " + img_name_final)

    create_outputs_json(img_names_list, output_data_dir, outputs_list, model_name)
    create_masks(outputs_list)


def create_outputs_json(img_names_list, output_data_dir, outputs_list, model_name):
    data_list = []
    images_list = []
    outputs_dict = {}
    for i in range(len(outputs_list)):
        instances = outputs_list[i]["instances"]
        img_size = instances.image_size
        fields = instances.get_fields()
        pred_boxes_numpy = fields['pred_boxes'].tensor.numpy()
        scores_numpy = fields['scores'].numpy()
        pred_classes_numpy = fields['pred_classes'].numpy()

        image = {"name": img_names_list[i],
                 "width": img_size[1], # TODO: zkontrolovat, zda nejsou prohozene width a height u obdelniku
                 "height": img_size[0],
                 "number_instances": len(pred_classes_numpy)}

        data = {"pred_boxes": pred_boxes_numpy.tolist(),
                "scores": scores_numpy.tolist(),
                "pred_classes": pred_classes_numpy.tolist()}

        data_list.append(data)
        images_list.append(image)

    version = "1.0"
    description = "Prediction outputs results"
    info_dictionary = COCO_json.get_info_dictionary(version, description, "")
    info_dictionary["model_name"] = model_name
    outputs_dict.update({"info": info_dictionary})
    outputs_dict.update({"images": images_list})
    outputs_dict.update({"outputs": data_list})

    # Creating .json file
    with open(os.path.join(output_data_dir, "processed", "outputs.json"), "w", encoding="utf-8") as f:
        json.dump(outputs_dict, f, ensure_ascii=False, indent=4)
        f.close()

def create_masks(outputs_list):
    path = create_directory("masks_prediction")
    for i in range(len(outputs_list)):
        instances = outputs_list[i]["instances"]
        masks = instances.get("pred_masks")
        masks_numpy = masks.numpy()
        plt.imshow(np.sum(masks_numpy.astype(int), axis=0))
        plt.savefig(os.path.join(path, "mask_" + str(i).zfill(4)))