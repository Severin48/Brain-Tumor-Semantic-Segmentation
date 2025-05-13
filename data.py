import os
import platform
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Set the root path depending on OS
OS_PREFIX = "D:/data" if platform.system() == "Windows" else ""
DATA_PATH = OS_PREFIX + "/kaggle/input/lgg-mri-segmentation/kaggle_3m/"
IMG_SIZE = 256

class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        mask_path = self.df.loc[idx, "mask_path"]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 0).astype(np.float32)  # binary mask

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def load_mri_dataframe(data_path=DATA_PATH):
    BASE_LEN = 89 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_ <-!!!43.tif)
    END_IMG_LEN = 4 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->.tif)
    END_MASK_LEN = 9 # (/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->_mask.tif)

    data_map = []
    for sub_dir_path in glob.glob(data_path + "*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = os.path.join(sub_dir_path, filename)
                data_map.extend([dirname, image_path])

    df = pd.DataFrame({"dirname": data_map[::2], "path": data_map[1::2]})
    df_imgs = df[~df['path'].str.contains("mask")]
    df_masks = df[df['path'].str.contains("mask")]

    imgs = sorted(df_imgs["path"].values, key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
    masks = sorted(df_masks["path"].values, key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

    df_final = pd.DataFrame({
        "patient": df_imgs.dirname.values,
        "image_path": imgs,
        "mask_path": masks
    })

    def positive_negative_diagnosis(mask_path):
        value = np.max(cv2.imread(mask_path))
        return 1 if value > 0 else 0  # Diagnosis Tumor = Positive if there is at least one non-black pixel i.e. a mask exists

    df_final["diagnosis"] = df_final["mask_path"].apply(positive_negative_diagnosis)
    return df_final


def get_dataloaders(df, batch_size=8, val_split=0.2, shuffle=True, transform=None):
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=48)

    train_dataset = MRIDataset(train_df, transform=transform)
    val_dataset = MRIDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
