import cv2
import os

def resize_images(folder_images, new_folder, size):

    for img in os.listdir(folder_images):
        orig_image = cv2.imread(os.path.join(folder_images, img), cv2.IMREAD_UNCHANGED)
        new_image_size = cv2.resize(orig_image, (size, size))
        cv2.imwrite(f"{new_folder}\\{img}", new_image_size)
