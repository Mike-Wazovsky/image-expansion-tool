import os
import shutil
import subprocess

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms.functional import resize

from PIL import Image


def remove_directory_content(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def download_and_process_data(path_to_download_script):
    if os.path.isdir("./data/raw") and len(os.listdir("./data/raw")) != 0:
        remove_directory_content("./data/raw")
        remove_directory_content("./data/processed/")

    subprocess.run(["chmod", "+x", path_to_download_script])
    subprocess.run([path_to_download_script])
    subprocess.run(["unzip", "./data/raw/dataset.zip", "-d", "./data/processed/"])


class SeagullDataset(Dataset):
    def __init__(self, path, transformations):
        self.filenames = []
        self.root = path
        self.transformations = transformations
        self.resizing_options = [transforms.InterpolationMode.NEAREST,
                                 transforms.InterpolationMode.NEAREST_EXACT,
                                 transforms.InterpolationMode.BILINEAR,
                                 transforms.InterpolationMode.BICUBIC]

        for (_, _, files) in os.walk(path):
            for file in files:
                # TODO: Проверять, чтобы картинка была более 512x512 px
                self.filenames.append(file)
            break

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(self.root + filename)

        Y_transformed = self.transformations(image)
        X_transformed = resize(img=Y_transformed,
                               size=Y_transformed.shape[1] // 2,
                               # interpolation=random.choice(self.resizing_options))
                               interpolation=transforms.InterpolationMode.BICUBIC)

        return X_transformed, Y_transformed


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=8):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return [train_loader, valid_loader, test_loader]
