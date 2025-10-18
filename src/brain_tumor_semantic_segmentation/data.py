import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.brain_tumor_semantic_segmentation.util import extract_index
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_PATH = os.path.expanduser(
    os.path.join("~", "kaggle", "input", "lgg-mri-segmentation", "kaggle_3m/")
)
IMG_SIZE = 256
STATS_FILE = "mri_dataset_stats.npy"


def _calculate_dataset_stats(data_path=DATA_PATH):
    """
    Helper function to calculate mean and std. Should only be run once.
    """
    image_paths = [p for p in glob.glob(os.path.join(data_path, "*", "*.tif")) if "mask" not in p]
    
    pixel_sum = np.zeros(3)
    pixel_sum_sq = np.zeros(3)
    pixel_count = 0
    
    print(f"Calculating dataset stats for {len(image_paths)} images... (this runs only once)")
    
    for img_path in tqdm(image_paths):
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image = image.astype(np.float32) / 255.0
        h, w, c = image.shape
        
        pixel_sum += image.sum(axis=(0, 1))
        pixel_sum_sq += (image ** 2).sum(axis=(0, 1))
        pixel_count += h * w
        
    mean = pixel_sum / pixel_count
    var = (pixel_sum_sq / pixel_count) - (mean ** 2)
    std = np.sqrt(var)
    
    # Return in RGB order as cv2 reads BGR
    return mean[::-1], std[::-1]

def get_or_calculate_dataset_stats(stats_file=STATS_FILE):
    """
    Loads dataset stats from a file or calculates and saves them if the file doesn't exist.
    """
    if os.path.exists(stats_file):
        print(f"Loading cached dataset stats from '{stats_file}'")
        stats = np.load(stats_file, allow_pickle=True).item()
        return stats['mean'], stats['std']
    else:
        mean, std = _calculate_dataset_stats()
        stats = {'mean': mean, 'std': std}
        np.save(stats_file, stats)
        print(f"Saved dataset stats to '{stats_file}'")
        return mean, std

DATASET_MEAN, DATASET_STD = get_or_calculate_dataset_stats()
print(f"Dataset Mean: {np.round(DATASET_MEAN, 4)}")
print(f"Dataset Std:  {np.round(DATASET_STD, 4)}")


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']           # already Tensor
            mask = augmented['mask'].unsqueeze(0) # (1, H, W)
        else:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.df.loc[idx, "mask_path"]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_label = 1 if np.max(mask) > 0 else 0

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, torch.tensor(binary_label, dtype=torch.long)

def validate_data_path(path: str):
    """Checks if the data path exists and raises a helpful error if not."""
    if not os.path.isdir(path):
        # Using an f-string to build a clear, multi-line error message
        error_message = (
            f"Data directory not found at the specified path: {path}\n"
            "Please ensure the data is downloaded and placed correctly.\n\n"
            "1. Download the dataset from: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation\n"
            "2. Extract the files to the expected location, which is structured like this:\n"
            f"  {DATA_PATH}\n"
            "   (The constant `DATA_PATH` in `src/brain_tumor_semantic_segmentation/data.py` points to this directory)"
        )
        raise FileNotFoundError(error_message)

def is_valid_image(img_path: str, threshold_percent: float = 100.0, threshold_intensity: int = 0) -> bool:
    """
    Checks if an image is not almost entirely dark.
    Returns True if the fraction of pixels below threshold_intensity is 
    less than threshold_percent.
    """
    # Read in grayscale as it's faster and sufficient for this check
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    
    # Calculate the ratio of dark pixels
    dark_pixels = np.sum(image < threshold_intensity)
    total_pixels = image.size
    dark_pixel_percentage = (dark_pixels / total_pixels) * 100
    
    # Return True if the image is NOT too dark
    return dark_pixel_percentage < threshold_percent
        
def load_mri_dataframe(data_path=DATA_PATH):
    validate_data_path(data_path)
    
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

    # Filter out images that are mostly black
    initial_count = len(df_final)
    df_final = df_final[df_final["image_path"].apply(is_valid_image)].reset_index(drop=True)
    filtered_count = initial_count - len(df_final)
    if filtered_count > 0:
        print(f"[Data] Filtered out {filtered_count}/{initial_count} images that were >90% black.")

    return df_final
    
    return df_final
    
def get_albu_augmentation(img_size=256):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(
            translate_percent=0.05, 
            rotate=10,         
            shear=5, 
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.2
        ),
        # auswirkung checken, siehe dataexploration!
        # A.ElasticTransform(alpha=0.5, sigma=10, p=0.05),
        A.Resize(img_size, img_size),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), 
        ToTensorV2()
    ])

def get_albu_img_transform(img_size=256):
    #macht aus Rohbildern gleiche Format wie train_transform (nur eben ohne Augmentation)! wichtig
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), 
        ToTensorV2()
    ])

def get_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 8,
    val_split: float = 0.2,
    shuffle: bool = True,
    augment: bool = False,
    omit_empty_masks: bool = False
) -> Tuple[DataLoader, DataLoader]:
    
    if omit_empty_masks:
        df = df[df["diagnosis"] == 1].reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=val_split, random_state=48, stratify=df["diagnosis"])

    train_transform = get_albu_augmentation(IMG_SIZE) if augment else get_albu_img_transform(IMG_SIZE)
    val_transform = get_albu_img_transform(IMG_SIZE)

    train_ds = MRIDataset(train_df, transform=train_transform)
    val_ds   = MRIDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print("[Data] Train images:", len(train_ds), "; Val images:", len(val_ds))
    return train_loader, val_loader

def get_dataloaders_from_dfs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    omit_empty_masks: bool = False  # not needed, since already filtered in hyperparameter_search.py
) -> Tuple[DataLoader, DataLoader]:


    train_transform = get_albu_augmentation(IMG_SIZE) if augment else get_albu_img_transform(IMG_SIZE)
    val_transform = get_albu_img_transform(IMG_SIZE)

    train_ds = MRIDataset(train_df, transform=train_transform)
    val_ds   = MRIDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print("[Data] Train images:", len(train_ds), "; Val images:", len(val_ds))
    return train_loader, val_loader

def get_dataloader_binarytransformed(
    df: pd.DataFrame,
    batch_size: int = 8,
    val_split: float = 0.15,
    shuffle: bool = True,
    augment: bool = False,
    omit_empty_masks: bool = False
) -> Tuple[DataLoader, DataLoader]:
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=48, stratify=df["diagnosis"])
    if omit_empty_masks:
        train_df = train_df[train_df.apply(
            lambda row: np.max(cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)) > 0, axis=1)
        ].reset_index(drop=True)
        val_df = val_df[val_df.apply(
            lambda row: np.max(cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)) > 0, axis=1)
        ].reset_index(drop=True)

    train_transform = get_albu_img_transform(IMG_SIZE) if not augment else get_albu_augmentation(IMG_SIZE)
    val_transform = get_albu_img_transform(IMG_SIZE)

    train_ds = MRIDatasetBinary(train_df, transform=train_transform)
    val_ds   = MRIDatasetBinary(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("[Data] (Binary) Train samples:", len(train_ds), "; Val samples:", len(val_ds))
    return train_loader, val_loader

def get_dataloaders_from_dfs_binary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates binary classification dataloaders from training and validation dataframes.
    """
    train_transform = get_albu_augmentation(IMG_SIZE) if augment else get_albu_img_transform(IMG_SIZE)
    val_transform = get_albu_img_transform(IMG_SIZE)

    train_ds = MRIDatasetBinary(train_df, transform=train_transform)
    val_ds   = MRIDatasetBinary(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print("[Data] (Binary) Train samples:", len(train_ds), "; Val samples:", len(val_ds))
    return train_loader, val_loader