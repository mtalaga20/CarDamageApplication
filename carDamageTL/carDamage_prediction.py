import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

# load the class label names from disk, one label per line
CLASS_NAMES = ['BG', 'Dent', 'Scratch', 'Crack', 'Glass Shatter', 'Broken Lamp', 'Flat Tire']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.80 

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="cardamage_poly_mask_rcnn_trained.h5", 
                   by_name=True)


count = 0
for img in os.listdir("C:\\Users\\mktal\\repos\\EECSProject\\CarDD_release\\CarDD_release\\CarDD_COCO\\images\\test2017\\"):
    count+=1
    if count % 10 == 0:
        image = cv2.imread("C:\\Users\\mktal\\repos\\EECSProject\\CarDD_release\\CarDD_release\\CarDD_COCO\\images\\test2017\\" + img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = model.detect([image], verbose=0)
        r = predictions[0]
    #images.append(image)
        #image, image_meta, gt_class_id, gt_bbox, gt_mask = mrcnn.model.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        mrcnn.visualize.display_instances(image=image, 
                                        boxes=r['rois'], 
                                        masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'])