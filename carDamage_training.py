import numpy as np
import os
import json
import skimage
import tensorflow as tf
from datetime import datetime
from numpy import zeros, asarray
from xml.etree import ElementTree
import imgaug.augmenters as iaa
import mrcnn.utils
import mrcnn.config
import mrcnn.model
from os import listdir

class DmgDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, subset):
        
        # we use add_class for each class in our dataset and assign numbers to them. 0 is background
        # self.add_class('source', 'class id', 'class name')
        self.add_class("damage", 1, "Dent")
        self.add_class("damage", 2, "Scratch")
        self.add_class("damage", 3, "Crack")
        self.add_class("damage", 4, "Glass Shatter")
        self.add_class("damage", 5, "Broken Lamp")
        self.add_class("damage", 6, "Flat Tire")
        
        #assert subset in ["train", "val"]
        #dataset_dir = os.path.join(dataset_dir, subset)
        
        # load annotations using json.load()
        annotations1 = json.load(open(dataset_dir + f"\\annotations\\instances_{subset}2017.json"))
        #annotations1 = json.load(open(os.path.join(dataset_dir, "via_project.json")))
        # convert annotations1 into a list
        #images = list(annotations1["images"].values())  
        #Pre-process
        for img in annotations1["images"]:
        # we only require the regions in the annotations
            id = img["id"]
            file_name = img["file_name"]
            annotations = [ann for ann in annotations1["annotations"] if ann["image_id"] == id]
            if len(annotations) > 0 and (id not in [52]):
                damages = [ann["category_id"] for ann in annotations]
                polygons = [{'all_points_x': ann["segmentation"][0][::2],
                          'all_points_y': ann["segmentation"][0][1::2]} for ann in annotations]
                #boxes = [{'all_points_x': ann["segmentation"][0][::2],
                #          'all_points_y': ann["segmentation"][0][1::2]} for ann in annotations]
                image_path = dataset_dir + f'\\images\\{subset}2017\\' + file_name
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                self.add_image(
                    "damage",
                    image_id=id,
                    path=image_path,
                    width = width,
                    height = height,
                    polygons=polygons,
                    num_ids = damages,
                )
    
    # this function calls on the extract_boxes method and is used to load a mask for each instance in an image
    # returns a boolean mask with following dimensions width * height * instances
    def load_mask(self, image_id):
        
        # info points to the current image_id
        info = self.image_info[image_id]
        
        # for cases when source is not damage
        if info["source"] != "damage":
            return super(self.__class__, self).load_mask(image_id)
        
        # get the class ids in an image
        num_ids = info['num_ids']
        
        
        
        # we create len(info["polygons"])(total number of polygons) number of masks of height 'h' and width 'w'
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        # we loop over all polygons and generate masks (polygon mask) and class id for each instance
        # masks can have any shape as we have used polygon for annotations
        # for example: if 2.jpg have four objects we will have following masks and class_ids
        # 000001100 000111000 000001110
        # 000111100 011100000 000001110
        # 000011111 011111000 000001110
        # 000000000 111100000 000000000
        #    1         2          3    <- class_ids
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1
            
        # return masks and class_ids as array
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
    
    # this functions takes the image_id and returns the path of the image
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "damage":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# define a configuration for the model
class DamageConfig(mrcnn.config.Config):
    # define the name of the configuration
    NAME = "damage"
    
    # number of classes (background + damge classes)
    NUM_CLASSES = 1+6
    
    # number of training steps per epoch
    STEPS_PER_EPOCH = 2815
    # learning rate and momentum
    LEARNING_RATE=0.0002
    LEARNING_MOMENTUM = 0.9
    
    # regularization penalty
    WEIGHT_DECAY = 0.0001
    
    # image size is controlled by this parameter
    IMAGE_MIN_DIM = 512
    
    # validation steps
    VALIDATION_STEPS = 810
    
    # number of Region of Interest generated per image
    Train_ROIs_Per_Image = 200
    
    # RPN Acnhor scales and ratios to find ROI
    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]

if __name__ == '__main__':
    # prepare train dataset.
    path = r"C:\Users\mktal\repos\EECSProject\CarDD_release\CarDD_release\CarDD_COCO"
    train_set = DmgDataset()
    # change the dataset 
    train_set.load_dataset(path, "train")
    train_set.prepare()

    # prepare validation/test dataset
    test_set = DmgDataset()
    test_set.load_dataset(path, "val")
    test_set.prepare()

    # load damage config
    config = DamageConfig()

    # define the model
    model = mrcnn.model.MaskRCNN(mode='training', model_dir='./', config=config)

    # load weights mscoco model weights
    weights_path = 'cardamage_poly_mask_rcnn_trained.h5'

    # load the model weights
    model.load_weights(weights_path, 
                    by_name=True, 
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    logdir = os.path.join(
        "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    # start the training of model
    # you can change epochs and layers (head or all)
    model.train(train_set, 
                test_set, 
                learning_rate=config.LEARNING_RATE, 
                epochs=4, 
                layers='heads',
                augmentation = iaa.Sometimes(5/6,iaa.OneOf([
                    iaa.Fliplr(1),
                    iaa.Flipud(1),
                    iaa.Affine(rotate=(-45, 45)),
                    iaa.Affine(rotate=(-90, 90)),
                    iaa.Affine(scale=(0.5, 1.5))
                    ])))

    model_path = 'cardamage_poly_mask_rcnn_trained3.h5'
    model.keras_model.save_weights(model_path)