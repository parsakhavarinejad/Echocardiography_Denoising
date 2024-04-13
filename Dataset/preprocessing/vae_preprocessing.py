import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DataGenerator(Dataset):
    """
    Custom data generator for VAE.

    Args:
        data (pd.DataFrame): DataFrame containing image paths.

        image_transform (transforms.Compose, optional): A sequence of
                          transformations to apply to the images. Defaults to None.
        input_size (tuple, optional): Desired size (height, width) for the
                                      resized images. Defaults to None.
        output_size (tuple, optional): Desired size (height, width) for the
                                      resized bounding boxes (if applicable).
                                      Defaults to None.
    """

    def __init__(self, data: pd.DataFrame, image_transform: transforms.Compose = None,
                 input_size: tuple = None, output_size: tuple = None):
        self.data = data
        self.image_transform = image_transform
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self) -> int:
        """Returns the length of the dataset (number of samples)."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a single sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The preprocessed image as a tensor.

        Raises:
            ValueError: If `image_path` column is missing in the DataFrame.
        """

        # Check for image path column existence
        if 'image_path' not in self.data.columns:
            raise ValueError("The DataFrame must contain an 'image_path' column.")

        image_path = self.data.iloc[idx]['image_path']
        image = Image.open(image_path).convert('L')  # Convert to grayscale (optional)

        # Apply image transformations if provided
        if self.image_transform:
            image = self.image_transform(image)

        return image

# # Example usage (assuming you have a DataFrame `data` containing image paths)
# train_data = DataGenerator(data, transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize for grayscale images
# ]))
