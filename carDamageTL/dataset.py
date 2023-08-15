import json
import skimage
import numpy as np

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class DamageDataset(mrcnn.utils.Dataset):

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
            if len(annotations) > 0 and id not in [90000]:
                damages = [ann["category_id"] for ann in annotations]
                boxes = [{'all_points_x': ann["segmentation"][0][::2],
                          'all_points_y': ann["segmentation"][0][1::2]} for ann in annotations]
                image_path = dataset_dir + f'\\images\\{subset}2017\\' + file_name
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                self.add_image(
                    "damage",
                    image_id=id,
                    image=image,
                    path=image_path,
                    width = width,
                    height = height,
                    polygons=boxes,
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
