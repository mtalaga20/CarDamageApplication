import numpy as np
from tqdm import tqdm
import mrcnn.config
import mrcnn.utils
import mrcnn.model
import mrcnn.visualize
import math

from carDamageTL.dataset import DamageDataset
from core import utilityFunctions

CLASSES = ['BG','Dent','Scratch','Crack','Glass Shatter', 'Broken Lamp','Flat Tire']


class PredictionConfig(mrcnn.config.Config):
    NAME = "damage"
    NUM_CLASSES = 1 + 6
    DETECTION_MIN_CONFIDENCE = 0.75
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# evaluate_model is used to calculate mean Average Precision of the model
def evaluate_model(dataset, model, cfg):
    APs = list()
    precision = list()
    recall = list()
    gt_tot = np.array([])
    pred_tot = np.array([])
    iou=0.35 #iou threshold
    class_ap = {'Dent': {'Count':0, 'AP': 0}, 'Scratch': {'Count':0, 'AP': 0}, 'Crack': {'Count':0, 'AP': 0},
             'Glass Shatter': {'Count':0, 'AP': 0}, 'Broken Lamp': {'Count':0, 'AP': 0}, 'Flat Tire': {'Count':0, 'AP': 0}}
    size_ap = {'S': {'Count':0, 'AP':0}, 'M': {'Count': 0, 'AP': 0}, 'L': {'Count':0, 'AP':0}}
    count = 0
    for image_id in tqdm(dataset.image_ids):
        count+=1
        image, image_meta, gt_class_id, gt_bbox, gt_mask = mrcnn.model.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        
        scaled_image = mrcnn.model.mold_image(image, cfg)

        sample = np.expand_dims(scaled_image, 0)

        prediction = model.detect(sample, verbose=0)

        r = prediction[0]

        gt, pred = utilityFunctions.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        gt_tot = np.append(gt_tot, gt)
        pred_tot = np.append(pred_tot, pred)
        #AP, _, _, _ = mrcnn.utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        img_AP = []
        img_prec = []
        img_rec = []
        for cls in np.unique(gt_class_id):
            for iou in [0.5]:
            #for iou in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                AP, precisions, recalls, _ = mrcnn.utils.compute_ap(gt_bbox, gt_class_id, gt_mask, cls, 
                                                    r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=iou)
                indices = np.where(gt_class_id==cls)
                for index in np.nditer(indices):
                    bbox = gt_bbox[index]
                    area = math.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    if area > 256:
                        size_ap['L']['Count'] += 1
                        size_ap['L']['AP'] += AP
                    elif area > 128:
                        size_ap['M']['Count'] += 1
                        size_ap['M']['AP'] += AP
                    else:
                        size_ap['S']['Count'] += 1
                        size_ap['S']['AP'] += AP
                if cls == 1:
                    class_ap['Dent']['Count'] += 1
                    class_ap['Dent']['AP'] += AP
                elif cls == 2:
                    class_ap['Scratch']['Count'] += 1
                    class_ap['Scratch']['AP'] += AP
                elif cls == 3:
                    class_ap['Crack']['Count'] += 1
                    class_ap['Crack']['AP'] += AP
                elif cls == 4:
                    class_ap['Glass Shatter']['Count'] += 1
                    class_ap['Glass Shatter']['AP'] += AP
                elif cls == 5:
                    class_ap['Broken Lamp']['Count'] += 1
                    class_ap['Broken Lamp']['AP'] += AP
                elif cls == 6:
                    class_ap['Flat Tire']['Count'] += 1
                    class_ap['Flat Tire']['AP'] += AP
                img_AP.append(AP)
                img_prec.append(np.mean(precisions))
                img_rec.append(np.mean(recalls))
        
        #Display results or ground truth
        #if count % 9 == 0:
        #    mrcnn.visualize.display_instances(image, r["rois"], r['masks'], r["class_ids"], dataset.class_names, r['scores'], title="Predictions")
        #    mrcnn.visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names, np.array([]), title="Ground Truth")

        APs.append(np.mean(img_AP))
        precision.append(np.mean(img_prec))
        recall.append(np.mean(img_rec))

    mAP = np.mean(APs)
    recall = np.mean(recall)
    precision = np.mean(precision)
    
    tp,fp,fn=utilityFunctions.plot_confusion_matrix_from_data(gt_tot,pred_tot,fz=17, figsize=(30,30), lw=0.5, columns=CLASSES)
    return mAP, precision, recall, class_ap, size_ap

#Eval
path = r"C:\Users\mktal\repos\EECSProject\CarDD_release\CarDD_release\CarDD_COCO"

#Evaluation
test_set = DamageDataset()
test_set.load_dataset(path, "test")
test_set.prepare()

cfg = PredictionConfig()
model = mrcnn.model.MaskRCNN(mode='inference', model_dir='./', config=cfg)
model_path = 'cardamage_poly_mask_rcnn_trained13.h5'
model.load_weights(model_path, by_name=True) 

# evaluate model on test dataset
test_mAP, test_precision, test_recall, class_ap, size_ap = evaluate_model(test_set, model, cfg)
print("Test mAP: %.4f" % test_mAP)
print("Test recall: %.4f" % test_recall)
print("Test precision: %.4f" % test_precision)

for cls in class_ap:
    print(f"Class {cls} has an AP of {100*(class_ap[cls]['AP']/class_ap[cls]['Count'])}")

print("")
for sz in size_ap:
    print(f"Class {sz} has an AP of {100*(size_ap[sz]['AP']/size_ap[sz]['Count'])}")