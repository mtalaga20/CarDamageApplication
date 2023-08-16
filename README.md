# CarDamageApplication

This project is for vehicle damage detection using the MaskR-CNN model by Matterport. It will eventually host other models for damage severity and location, along with a pipeline for using the models together for damage cost prediction.

## Installation
1. Clone the repository
2. Install dependencies - `pip install -r requirements.txt`
3. The dataset and pre-trained weights will need to be installed before running the code.
- The dataset is available [here](https://cardd-ustc.github.io/) and can be added to the local project folder.
- The pre-trained weights (mask_rcnn_coco.h5) can be downloaded from [here](https://github.com/matterport/Mask_RCNN/releases) and placed in the project folder too.

## Usage

1. Replace the path variable in the carDamage_evaluation.py and carDamage_training.py files with the path to your data.
2. Train - `python carDamage_training.py`
3. Test - `python carDamage_evaluation.py`
