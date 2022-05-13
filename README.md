# Web_App_PyWebIO
## Description
The main goal of this computer vision application is to detect cell nuclei in microscopic histological images. 
It is based on Detectron2 framework, so for the detection it uses deep neural network. User also can train his own model through this application, too.

The application was written in Python and it is powered by PyWebIO module.

## How the app works
![alt text](https://github.com/janburian/Web_App_PyWebIO/blob/main/schema2.png?raw=true)

* Step 1: User uploads the data in .czi format (with or without the annotations)
* Step 2: User chooses the option if he wants to predict the data or train the new model 

Predict
* Step 3: Now, there is option to delete some of the available models. This step is optional and can be skipped. 
* Step 4: User chooses the model which should be used to predict the data. 
* Step 5: Finally data with the predictions are visualized. User can download the results. 

Train 
(data with the annotations are needed in this case)
* Step 3: User can upload annotated files in .czi format to create the test/validation dataset, otherwise the test dataset will be created automatically. 
* Step 4: User writes the category of the dataset. 
* Step 5: Training dataset is visualized. And after a while the results can be downloaded. The trained model is added to available models for prediction. Name of trained model is defined by the user.    


## Installation of important modules to run the app
* conda (installation: https://docs.conda.io/en/latest/miniconda.html)
* scaffan (installation steps: https://github.com/mjirik/scaffan/blob/master/README.md)
* detectron2 (installation steps: https://github.com/mjirik/tutorials/blob/main/detectron_windows/readme.md)
* PyWebIO (pip3 install -U pywebio)

All of these modules need to be installed in the same Python environment. 

## Sofware for creating annotations and processing microscopic images
Zeiss ZEN (https://www.zeiss.com/microscopy/int/products/microscope-software/zen-lite.html)
