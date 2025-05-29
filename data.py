import os
import platform
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
import re
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from util import extract_index

OS_PREFIX = "D:/data/" if platform.system() == "Windows" else ""
DATA_PATH = OS_PREFIX + "kaggle/input/lgg-mri-segmentation/kaggle_3m/"
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

class MRIDatasetBinary(Dataset):
    """
    Dataset that transforms the segmentation data into binary labels.
    For each sample, if the mask contains any tumor (i.e. any pixel > 0),
    the label is 1; otherwise it is 0.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        
        image = cv2.imread(img_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        

        mask_path = self.df.loc[idx, "mask_path"]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
   
        binary_label = 1 if np.max(mask) > 0 else 0
        
    
        return image, torch.tensor(binary_label, dtype=torch.long)
        
def load_mri_dataframe(data_path=DATA_PATH):
    data_map = []
    for sub_dir_path in glob.glob(data_path + "*"):
        if os.path.isdir(sub_dir_path):
            dirname = os.path.basename(sub_dir_path)
            for filename in os.listdir(sub_dir_path):

                if not filename.lower().endswith(".tif"):
                    continue

                full = os.path.join(sub_dir_path, filename)
                data_map.append((dirname, full))

    df = pd.DataFrame(data_map, columns=["patient", "path"])
    
    df_imgs = df[~df["path"].str.contains("mask")]
    df_masks = df[df["path"].str.contains("mask")]

    imgs = sorted(df_imgs["path"].tolist(), key=extract_index)
    masks = sorted(df_masks["path"].tolist(), key=extract_index)

    df_final = pd.DataFrame({
        "patient": [os.path.basename(os.path.dirname(p)) for p in imgs],
        "image_path": imgs,
        "mask_path": masks
    })

    # Add binary diagnosis column
    def positive_negative_diagnosis(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return 1 if m.max() > 0 else 0

    df_final["diagnosis"] = df_final["mask_path"].apply(positive_negative_diagnosis)
    return df_final

def get_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 8,
    val_split: float = 0.2,
    shuffle: bool = True,
    transform: transforms.Compose | None = None,
    omit_empty_masks: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
    """Return *(train_loader, val_loader)*.

    If omit_empty_masks is True, rows whose masks are completely black
    (diagnosis == 0) are discarded after the train/val split (so each
    split is filtered independently and keeps its original size ratio).
    """

    train_df, val_df = train_test_split(df, test_size=val_split, random_state=48)

    if omit_empty_masks:
        train_df = train_df[train_df["diagnosis"] == 1].reset_index(drop=True)
        val_df   = val_df[val_df["diagnosis"] == 1].reset_index(drop=True)

    train_ds = MRIDataset(train_df, transform=transform)
    val_ds   = MRIDataset(val_df,   transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print("[Data] Train images:", len(train_ds), "; Val images:", len(val_ds))
    return train_loader, val_loader

def get_dataloader_binarytransformed(
    df: pd.DataFrame,
    batch_size: int = 8,
    val_split: float = 0.2,
    shuffle: bool = True,
    transform: transforms.Compose | None = None,
    omit_empty_masks: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns (train_loader, val_loader) for the binary classification
    task. It transforms the segmentation mask to a binary label (tumor/no tumor).
    
    Parameter omit_empty_masks can be used to filter out samples that do not exhibit a tumor.
    """
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=48)

    if omit_empty_masks:
        train_df = train_df[train_df.apply(
            lambda row: np.max(cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)) > 0, axis=1)
        ].reset_index(drop=True)
        val_df = val_df[val_df.apply(
            lambda row: np.max(cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)) > 0, axis=1)
        ].reset_index(drop=True)

    train_ds = MRIDatasetBinary(train_df, transform=transform)
    val_ds   = MRIDatasetBinary(val_df, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("[Data] (Binary) Train samples:", len(train_ds), "; Val samples:", len(val_ds))
    return train_loader, val_loader

