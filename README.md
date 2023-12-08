# fabric_defect_image_segmentation
This repository holds the code to use a UNet for defect segmentation in a data provided by: https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database

# Setting up your environment
The code in this repository use the following repositories. Please ensure your python environment is installed with the following repos (and versions):
- skimage version == 0.19.3  -- Note that this code will not work with skimage version 0.21.0 (e.g. in current (11/13/23) Kaggle environment)
- matplotlib version >= 3.7.1
- numpy version >= 1.23.5
- scipy version >= 1.11.3
- h5py version >= 3.9.0
- tensorflow version >= 2.14.0
- keras version >= 2.14.0
- pandas version >= 1.5.3

On a system with Python 3 pre-installed, an environment with these packages can be installed with the following bash commands:
~~~~
# Using venv (Python 3.3 and newer)
python3 -m venv fabric_defect_image_segmentation

# Activate the virtual environment
source fabric_defect_image_segmentation/bin/activate  # On Windows, use `fabric_defect_image_segmentation\Scripts\activate`

# Now, install the specific versions of the required packages
pip install scikit-image==0.19.3 matplotlib==3.7.1 numpy==1.23.5 scipy==1.11.3 h5py==3.9.0 tensorflow==2.14.0 pandas==1.5.3
~~~~

# Clone repo to local environment

To get started with the code in this repository, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/mushroom-matthew/fabric_defect_image_segmentation.git
    ```
    *Please note that the default repo name is the same as was used above for the virtual environment. This code will not work if you run it in the parent directory that contains your environment directory.*
    
2. Now, you can explore the code within the module.

If you encounter any issues or have questions, feel free to reach out or open an issue on GitHub.

# Running inference on an image (or directory of images)

To run the inference pipeline and generate a printed defect report, please run the following line of code from outside the `fabric_defect_image_segmentation` module directory.

*v1 model - Not currently preloaded, please train your own model (see steps below) to use a v1 model*
~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.defect_segmentation --model {/absolute/path/to/}fabric_defect_image_segmentation/models/pretrained_model.h5 --image {/absolute/path/to/image/or/directory/of/images/such/as/}fabric_defect_image_segmentation/data/sample_data/
~~~~

*v2 model*
~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.defect_segmentation --model {/absolute/path/to/}fabric_defect_image_segmentation/models/30epoch_final_model_weights_v2.h5 --image {/absolute/path/to/image/or/directory/of/images/such/as/}fabric_defect_image_segmentation/data/sample_data/
~~~~

This script can take additional arguments which include save options for masks, logits, and csv reports for found defects.

~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.defect_segmentation --help

usage: defect_segmentation.py [-h] --model MODEL --image IMAGE [--grain GRAIN] [--post-process POST_PROCESS] [--save-label-mask] [--save-logits-map] [--save-reports] [--results-folder RESULTS_FOLDER]
                              [--reports-folder REPORTS_FOLDER]

Defect Segmentation Script

options:
  -h, --help            show this help message and exit
  --model MODEL         Absolute path to the model weights file
  --image IMAGE         Absolute path to the input image file or directory
  --grain GRAIN         Stride of patch extraction (smaller grain --> more patches --> more accurate --> longer runtime)
  --post-process POST_PROCESS
                        Post-processing method (argmax or prob_thresh)
  --save-label-mask     Save mask images (post-process labeled)
  --save-logits-map     Save logit images (softmax layer output)
  --save-reports        Save reports
  --results-folder RESULTS_FOLDER
                        Absolute path to the results output folder
  --reports-folder REPORTS_FOLDER
                        Absolute path to the reports output folder
~~~~

*Please note that there are two models that are currently supported. Unfortunately, both models are known to be susceptible to false positives (in particular). Reducing the `--grain` input argument can help for those false positives related to edge effects of patch operations. Another known source of false positives seems related to illumination variations across samples. The v2 model is a bit better for false positives, but not perfect.*

*An additional option --post_process has also been added in an attempt to reduce the false positive rate.*

# Evaluating the segmentation pipeline performance against ground-truth masks

To evaluate the inference pipeline against a known ground-truth mask and generate a printed performance report, please run the following line of code from outside the `fabric_defect_image_segmentation` module directory.

*v1 model - Not currently preloaded, please train your own model (see steps below) to use a v1 model*
~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.assess_segmentation_performance --model {/absolute/path/to/}fabric_defect_image_segmentation/models/pretrained_model.h5 --image~ {/absolute/path/to/image/or/directory/of/images/such/as/}fabric_defect_image_segmentation/data/sample_data/ --mask {/absolute/path/to/image/or/directory/of/images/such/as/}fabric_defect_image_segmentation/data/sample_masks/
~~~~

*v2 model*
~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.assess_segmentation_performance --model {/absolute/path/to/}fabric_defect_image_segmentation/models/30epoch_final_model_weights_v2.h5 --image {/absolute/path/to/image/or/directory/of/images/such/as/}fabric_defect_image_segmentation/data/sample_data/ --mask {/absolute/path/to/image/or/directory/of/images/such/as/}fabric_defect_image_segmentation/data/sample_masks/
~~~~

This script can take additional arguments which include save options for masks, logits, and csv reports for found defects. Please see `--help` for more info.

# Extracting image/mask data from raw directories for training the resident UNet

The code contained in the module script `prepare_training_data` can be used to prep the data from raw images as supplied in the [kaggle database](https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database). There is an assumption that the data has been downloaded and that the images are arranged in the following structure:

~~~~bash
ParentDirectory
├── Defect_images
├── Mask_images
└── NODefect_images
~~~~

*The code is case sensitive to the 'NODefect' string in particular.*

Once the data is downloaded to your system and has the above structure, the following command can be run to produce the refined training and validation data. *Refined, in this case, means maximizing the representation of the underrepresented classes in the training dataset.*

~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.prepare_training_data --preprocess v2 --image-dirs {/absolute/path/to/training/Defect_images} {/absolute/path/to/training/NODefect_images} --mask-dirs {/absolute/path/to/Mask_images} --training-folder {/absolute/path/to/}data/training_data/
~~~~

The initial patch extraction or the dataset refinement can be achieved by adding `--skip-extract` or `--skip-refine` flags, respectively. Other options can be seen by using `--help`.

# Training the UNet in Segmentation workflow (training data is not supplied as it is >100MB, please run previous step to generate your training data locally)

Once training and validation datasets have been created (some are packaged with this repo in data/training_data), the resident UNet can be trained with the following command.

~~~~bash
python3 -m fabric_defect_image_segmentation.scripts.train_resident_unet --preprocess v2 --data-path {/absolute/path/to/model/save/directory/such/as}/fabric_defect_image_segmentation/models/ --training-data {/absolute/path/to/training_data/such/as}/training_data/training_patches_pv2.h5 --validation-data {/absolute/path/to/training_data/such/as}/training_data/validation_patches_pv2.h5
~~~~

See `--help` for additional options including an option for initializing a pretrained model for tuning (`--starting-model`). *Please note that v2 models require the 'v2' string in their naming convention.*