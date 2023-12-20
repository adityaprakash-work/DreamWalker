# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last modified: 20-12-2023

# --Needed functionalities
# - 1. Kaggle credential patching. Currently using vanilla opendatasets' method.

# ---DEPENDENCIES---------------------------------------------------------------
import os
import glob
from opendatasets import download
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ---TRANSFORMS-----------------------------------------------------------------
def default_transform(shape=(128, 128)):
    return transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


# ---DATASETS-------------------------------------------------------------------
class ImageStream(Dataset):
    def __init__(self, dir, ext="jpg", transform=default_transform()):
        self.dir = dir
        self.transform = transform
        pattern = os.path.join(self.dir, "**", "*." + ext)
        self.files = glob.glob(pattern, recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = self.transform(img)
        return img, 0  # 0 for compatibility with other datasets, [label]


# ---DATALOADERS----------------------------------------------------------------
def get_loaders(
    dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    return_valid=False,
    valid_split=0.2,
):
    if return_valid:
        vsize = int(valid_split * len(dataset))
        tsize = len(dataset) - vsize
        tdataset, vdataset = random_split(dataset, [tsize, vsize])
        tloader = DataLoader(
            tdataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        vloader = DataLoader(
            vdataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return tloader, vloader
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return loader


# ---END------------------------------------------------------------------------
