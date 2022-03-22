from pathlib import Path
import sys
import skimage.io
import os

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image


def export_czi_annotations_to_jpg(path_annotations, annotation_name, path_images):
    """
    Creates image dataset from .czi annotations

    Parameters:
    -----------
    path_annotations : Path
                       path to .czi files
    annotation_name : str
                      wanted name of the annotations (e.g. ann, annotation, a,...)
    path_images : Path
                  path to directory where the images will be saved
    """
    index = 0
    while True:
        fn_str = (
            str(path_annotations)
            + "\\"
            + annotation_name
            + str(index).zfill(4)
            + ".czi"
        )
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


# Loading .czi annotations
path_annotations = Path(
    r"H:\BP\data\dataset_maxi\czi_files_predict"
)  # path to main directory, that is where .czi files are
path_images = Path(
    r"H:\BP\datasets\dataset_maxi\dataset_maxi_prediction"
)  # path to directory, where the images will be saved
annotation_name = "annotation"

export_czi_annotations_to_jpg(path_annotations, annotation_name, path_images)
