from PIL import Image
import os

def crop_image(file):
    center_horizontal = file.shape[2]
    center_vertical = file.shape[1]

    return file[:, center_vertical // 6:center_vertical * 3 // 4, center_horizontal // 4:center_horizontal * 3 // 4, 0]


def save_cropped_image(image_matrix, destination):
    if image_matrix.dtype != 'uint8':
        image_matrix = (image_matrix * 255).astype('uint8')

    for y in range(0, image_matrix.shape[0], 5):

        image = Image.fromarray(image_matrix[y, :, :])
        image.save(os.path.join(destination, f"image_{y}.jpg"))
