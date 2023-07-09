import json
import pandas as pd
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as tf
from torchvision.datasets import ImageFolder
from tqdm import trange
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from evaluation import cross_entropy
from preprocess import preprocess
from preprocess import DamageDataset
from classifiers import MultilabelClassifier

if __name__ == '__main__':
    #Setup and Pre-processing
    #------------------------------------------------------------------------------------------------------------------
    #Manual Section (inputs)
    debug = True
    epochs = 6
    learning_rate = 0.01
    batch_size = 16
    backbone = "Resnet34"
    loss_function = "BCEWithLogitsLoss"
    transform = tf.Compose([   
            tf.Resize(size=(224,224)),
            tf.ToTensor()
        ])
    #------------------------------------------------------------------------------------------------------------------
    type = "test" if debug else "train"
    json_path = json_path = r"CarDD_release\CarDD_release\CarDD_COCO"
    data = preprocess(json_path)
    train = data[type] #NOTE
    translation_dict = dict(zip(train["filenames"],list(zip(train["samples"][0],train["samples"][1],train["samples"][2],train["samples"][3],train["samples"][4],train["samples"][5]))))
    train_set = DamageDataset(json_path + f"\\images\\{type}2017", transform=transform, translation_dict=translation_dict)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultilabelClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Training
    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in trange(epochs):
        losses = []
        for sample in tqdm(train_loader): 
            img = sample["image"].to(device)
            predictions = model(img)
            #labls = list(sample["labels"].values())
            loss = 0
            for key in predictions:
                loss += loss_function(torch.flatten(predictions[key]), torch.as_tensor([float(lbl) for lbl in sample['labels'][key]]).to(device))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"\nCP Loss at epoch {epoch+1} - {torch.tensor(losses).mean().item()}")
    print()
    #Validation and Testing
    test = data["val"]
    translation_dict = dict(zip(test["filenames"],list(zip(test["samples"][0],test["samples"][1],test["samples"][2],test["samples"][3],test["samples"][4],test["samples"][5]))))
    test_set = DamageDataset(json_path + f"\\images\\val2017", transform=transform, translation_dict=translation_dict)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=16)
    
    with torch.no_grad():
        order = ["dent", "scratch", "crack", "glass shatter",  "lamp broken", "tire flat"]
        confusion_matrices = [[0,0,0,0] for i in range(len(order))]
        total = [0 for i in range(len(order))]
        TP, TN, FN, FP = 0, 0, 0, 0

        for sample in tqdm(test_loader):
            img = sample["image"].to(device)
            predictions = model(img)
            labels = [sample['labels'][label].to(device) for label in sample['labels']]

            for i,out in enumerate(predictions):
                _, predicted = torch.max(predictions[out],1)
                for j in range(batch_size):
                    label = labels[i][j]
                    prediction = predicted[j]
                    if(label == 1 and prediction == 1):
                        TP += 1
                        confusion_matrices[i][0] += 1
                    elif(label==1):
                        FN +=1
                        confusion_matrices[i][1] += 1
                    elif(prediction==1):
                        FP +=1
                        confusion_matrices[i][2] += 1
                    else:
                        TN +=1
                        confusion_matrices[i][3] += 1
                    total[i] += 1
        for i in range(len(order)):
            confusion_matrix = confusion_matrices[i]
            accuracy = (confusion_matrix[0]+confusion_matrix[3]) / total[i]
            try: precision = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[2]) 
            except ZeroDivisionError: precision = 0 
            try: recall = confusion_matrix[0] / (confusion_matrix[0]+confusion_matrix[1])
            except ZeroDivisionError: recall = 0
            print("---------------------------------")
            print(f"""Feature - {order[i]}\n
                  Accuracy: {accuracy}\n
                   Precision: {precision}\n
                    Recall: {recall}\n\n""")
            confusion_matrices[i] = [accuracy, precision, recall]
        
        accuracy = np.mean([confusion_matrices[i][0] for i in range(len(confusion_matrices))])
        try: precision = np.mean([confusion_matrices[i][1] for i in range(len(confusion_matrices))])
        except: ZeroDivisionError: precision = "N/A"
        try: recall = np.mean([confusion_matrices[i][2] for i in range(len(confusion_matrices))])
        except: ZeroDivisionError: recall = "N/A"
        print("---------------------------------")
        print(f"""Total\n
                  Accuracy: {accuracy}\n
                   Precision: {precision}\n
                    Recall: {recall}""")
        

        results = [{"epochs": epochs, "learning_rate": learning_rate, "method": backbone, "loss_function": loss_function,
                    "accuracy" : accuracy, "precision": precision, "recall": recall,
                   "dent_accuracy": confusion_matrices[0][0], "dent_precision": confusion_matrices[0][1], "dent_recall": confusion_matrices[0][2],
                    "scratch_accuracy": confusion_matrices[1][0], "scratch_precision": confusion_matrices[1][1], "scratch_recall": confusion_matrices[1][2],
                    "crack_accuracy": confusion_matrices[2][0], "crack_precision": confusion_matrices[2][1], "crack_recall": confusion_matrices[2][2],
                    "glass_shatter_accuracy": confusion_matrices[3][0], "glass_shatter_precision": confusion_matrices[3][1], "crack_recall": confusion_matrices[3][2],
                    "lamp_broken_accuracy": confusion_matrices[4][0], "lamp_broken_precision": confusion_matrices[4][1], "crack_recall": confusion_matrices[4][2],
                    "tire_flat_accuracy": confusion_matrices[5][0], "tire_flat_precision": confusion_matrices[5][1], "crack_recall": confusion_matrices[5][2]}]
        df = pd.read_csv("results.csv")
        results_df = pd.concat([df, pd.DataFrame.from_dict(results)], ignore_index=True)
        results_df.to_csv("results.csv", index=False)

