from pydicom import dcmread
import os
import glob
from utils import crop_image, save_cropped_image

def dicom_extract(directory):
    dataset_list = glob.glob(os.path.join(directory, "*"))
    for dicom in dataset_list:
        dir = dicom.split('\\')[-1]
        path = f"Extracted/{dir}"
        if not os.path.exists(path):
            os.makedirs(path)

        echo = dcmread(dicom).pixel_array
        cropped = crop_image(echo)
        save_cropped_image(cropped, path)
        print(f"{dir} has been cropped, extracted and saved")


if __name__ == "__main__":
    dicom_extract("DICOM/")

