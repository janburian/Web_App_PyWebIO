# **Usage of Detectron2**
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# import some common libraries
import numpy as np
import cv2
import glob
import os
from pathlib import Path

setup_logger()

def create_directory(directory_name):
    files_directory = os.path.join(Path(__file__).parent, directory_name)
    if not os.path.exists(files_directory):
        os.makedirs(files_directory)
    return files_directory


def register_coco_instances(input_data_path):
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("cells_training", {}, os.path.join(input_data_path, "trainval.json"), os.path.join(input_data_path, "images"))
    cells_metadata = MetadataCatalog.get("cells_training")
    dataset_dicts = DatasetCatalog.get("cells_training")

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
    cfg.merge_from_file(r"C:\Users\janbu\miniconda3\envs\scaffan_2\Lib\site-packages\detectron2\model_zoo\configs\COCO-InstanceSegmentation"
                        "\mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("cells_training",)
    #cfg.DATASETS.TEST = ("cells_validation",)  # no metrics implemented for this dataset, validation dataset
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.MODEL.WEIGHTS = str(Path(__file__).parent / "models/model_final.pth") # TODO: choosing model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00005
    cfg.SOLVER.MAX_ITER = 50
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (cells nuclei)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(os.path.abspath(cfg.OUTPUT_DIR))
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



def predict(input_data_dir, output_data_dir):
    cfg = get_cfg()
    cfg.merge_from_file(
        r"C:\Users\janbu\miniconda3\envs\scaffan_2\Lib\site-packages\detectron2\model_zoo\configs\COCO-InstanceSegmentation"
        "\mask_rcnn_R_50_FPN_3x.yaml")

    #cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(), "models", "model_final.pth")
    cfg.MODEL.WEIGHTS = str(Path(__file__).parent / "models/model_final.pth") # TODO: choosing model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("cells_training",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (cells nuclei)
    cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    picture_predictions = glob.glob(str(input_data_dir) + "/*.jpg")
    number_pictures_predictions = len(picture_predictions)

    print("Number of pictures for prediction: " + str(number_pictures_predictions))
    index = 0
    for d in range(number_pictures_predictions):
        index_str = str(index)
        # print("TEST: " + str(input_data_dir_predict) + "/" + index_str.zfill(4) + ".jpg")
        print(os.path.join(input_data_dir, index_str.zfill(4), ".jpg"))
        im = cv2.imread(os.path.join(input_data_dir, index_str.zfill(4) + ".jpg"))
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       #metadata=cells_metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        Path(os.path.join(output_data_dir, "processed", "vis_predictions")).mkdir(parents=True, exist_ok=True)
        img_name_final = "pic_pred_" + index_str.zfill(4) + ".jpg"
        index += 1
        if not cv2.imwrite(os.path.join(output_data_dir, "processed", "vis_predictions", img_name_final), v.get_image()[:, :, ::-1]):
            raise Exception("Could not write image: " + img_name_final)

