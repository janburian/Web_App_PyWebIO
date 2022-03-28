# Detectron2 - cells.ipynb
# Original file is located at https://colab.research.google.com/drive/1rCLduoeFC_UKN5MEKF77nvPVkbxOzt1x

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

'''
Obtaining directories and checking if exist
'''
scratchdir = os.getenv('SCRATCHDIR', ".")
print(Path(scratchdir).exists())
print(Path(scratchdir))

input_data_dir_train = Path(scratchdir) / 'data/orig/cells/dataset_training'
print(Path(input_data_dir_train).exists())
print(Path(input_data_dir_train))

input_data_dir_validate = Path(scratchdir) / 'data/orig/cells/dataset_validation'
print(Path(input_data_dir_validate).exists())
print(Path(input_data_dir_validate))

input_data_dir_predict = Path(scratchdir) / 'data/orig/cells/dataset_prediction'
print(Path(input_data_dir_predict).exists())
print(Path(input_data_dir_predict))

output_dir = Path(scratchdir) / 'data/processed/'
print(Path(output_dir).exists())
print(Path(output_dir))

logs_dir = Path(scratchdir) / 'data'

# Get the list of all files and directories
path = str(input_data_dir_train)
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

# prints all files
print(dir_list)


''' 
Registering coco instances and getting metadata 
'''
from detectron2.data.datasets import register_coco_instances

register_coco_instances("cells_training", {}, str(input_data_dir_train / "trainval.json"), str(input_data_dir_train / "images"))
register_coco_instances("cells_validation", {}, str(input_data_dir_validate / "trainval.json"), str(input_data_dir_validate / "images"))

cells_metadata = MetadataCatalog.get("cells_training")
dataset_dicts = DatasetCatalog.get("cells_training")

print()
print(dataset_dicts)
print()


'''
Checking annotated pictures
'''
for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cells_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d)
    (output_dir / "vis_train").mkdir(parents=True, exist_ok=True)
    print(Path(output_dir / "vis_train").exists())
    print(Path((output_dir / "vis_train")))
    img_name_final = os.path.basename(d["file_name"])
    print(str(output_dir) + "/" + "vis_train" + "/" + img_name_final)
    if not cv2.imwrite(str(output_dir) + "/" + "vis_train" + "/" + img_name_final, vis.get_image()[:, :, ::-1]):
        raise Exception("Could not write image: " + img_name_final)


'''
Training 
'''
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# Parameters
cfg = get_cfg()
cfg.merge_from_file("/auto/plzen1/home/jburian/extern/detectron2/configs/COCO-InstanceSegmentation"
                    "/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("cells_training", )
cfg.DATASETS.TEST = ("cells_validation", )  # no metrics implemented for this dataset, validation dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.000002
cfg.SOLVER.MAX_ITER = 120000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (cells nuclei)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(os.path.abspath(cfg.OUTPUT_DIR))

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

print("Obsah adresare output_dir: " + str(list(Path(output_dir).glob("**/*"))))
print("Obsah adresare inputdir: " + str(list(Path(input_data_dir_train).glob("**/*"))))

trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATASETS.TEST = ("cells_training",)
predictor = DefaultPredictor(cfg)


'''
Predictions in pictures
'''
from detectron2.utils.visualizer import ColorMode

picture_predictions = glob.glob(str(input_data_dir_predict) + "/*.jpg")
number_pictures_predictions = len(picture_predictions)

print()
print("Number of pictures for prediction: " + str(number_pictures_predictions))

index = 0
for d in range(number_pictures_predictions):
    index_str = str(index)
    #print("TEST: " + str(input_data_dir_predict) + "/" + index_str.zfill(4) + ".jpg")
    im = cv2.imread(str(input_data_dir_predict) + "/" + index_str.zfill(4) + ".jpg")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=cells_metadata,
                   scale=3,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    (output_dir / "vis_predictions").mkdir(parents=True, exist_ok=True)
    img_name_final = "pic_pred_" + index_str.zfill(4) + ".jpg"
    index += 1
    if not cv2.imwrite(str(output_dir) + "/" + "vis_predictions" + "/" + img_name_final, v.get_image()[:, :, ::-1]):
        raise Exception("Could not write image: " + img_name_final)

