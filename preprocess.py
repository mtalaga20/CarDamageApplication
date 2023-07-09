import json
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as tf
from PIL import Image
from os import listdir
from torch.utils.data import Dataset

class DamageDataset(Dataset):
    def __init__(self,img_dir,transform,translation_dict):
        self.img_dir = img_dir
        self.folder = [x for x in listdir(img_dir) if x != 'annotations']
        self.transform = transform
        self.translation_dict = translation_dict

    def __len__(self):
        return len(self.folder)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.folder[idx])
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)

        label1 = self.translation_dict[self.folder[idx]][0]
        label2 = self.translation_dict[self.folder[idx]][1]
        label3 = self.translation_dict[self.folder[idx]][2]
        label4 = self.translation_dict[self.folder[idx]][3]
        label5 = self.translation_dict[self.folder[idx]][4]
        label6 = self.translation_dict[self.folder[idx]][5]

        sample = {'image':img, 
                  'labels': 
                        {"dent" : label1,
                        "scratch": label2,
                        "crack": label3,
                        "glass_shatter": label4,
                        "lamp_broken": label5,
                        "tire_flat": label6
                        }
                    }
        return sample   
    
def preprocess(json_path: str):
    transform = tf.Compose([   
        tf.Resize(size=(224,224)),
        tf.ToTensor()
    ])
    groups = ["train", "val", "test"]
    data = {}
    for group in groups:
        json_data = json.load(open(json_path + f"/annotations/instances_{group}2017.json"))
        filenames = []
        dent_list, scratch_list, crack_list, glass_shatter_list, lamp_broken_list, tire_flat_list = [], [], [], [], [], []
        for img in json_data["images"]:
            file_name = img["file_name"]
            filenames.append(file_name)
            id = img["id"]
            #image = transform(Image.open(json_path + f"\\{group}2017\\{file_name}").convert('RGB'))
            annotations = [ann["category_id"] for ann in json_data["annotations"] if ann["image_id"] == id]
            '''
            labels = {"dent" : (1 if 1 in annotations else 0),
                    "scratch": (1 if 2 in annotations else 0),
                    "crack": (1 if 3 in annotations else 0),
                    "glass shatter": (1 if 4 in annotations else 0),
                    "lamp broken": (1 if 5 in annotations else 0),
                    "tire flat": (1 if 6 in annotations else 0)}
            '''
            dent_list.append([1 if 1 in annotations else 0])
            scratch_list.append([1 if 2 in annotations else 0])
            crack_list.append([1 if 3 in annotations else 0])
            glass_shatter_list.append([1 if 4 in annotations else 0])
            lamp_broken_list.append([1 if 5 in annotations else 0])
            tire_flat_list.append([1 if 6 in annotations else 0])
        dent_list = np.asarray(dent_list)
        dent_list = torch.from_numpy(dent_list.astype('int'))
        scratch_list = np.asarray(scratch_list)
        scratch_list = torch.from_numpy(scratch_list.astype('int'))
        crack_list = np.asarray(crack_list)
        crack_list = torch.from_numpy(crack_list.astype('int'))
        glass_shatter_list = np.asarray(glass_shatter_list)
        glass_shatter_list = torch.from_numpy(glass_shatter_list.astype('int'))
        lamp_broken_list = np.asarray(lamp_broken_list)
        lamp_broken_list = torch.from_numpy(lamp_broken_list.astype('int'))
        tire_flat_list = np.asarray(tire_flat_list)
        tire_flat_list = torch.from_numpy(tire_flat_list.astype('int'))
        samples = [dent_list, scratch_list, crack_list, glass_shatter_list, lamp_broken_list, tire_flat_list]
            #samples.append({"file_name": file_name, "image": image, "labels": labels})
        data[group] = {"samples":samples, "filenames":filenames}
    return data


