import cv2
import numpy as np
import os

input_dir = 'images'
output_dir = 'output_spectrum'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_image(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    cv2.imwrite(output_path, magnitude_spectrum)

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, 'spectrum_' + filename)
        process_image(image_path, output_path)

print("Process complete and save all spectrum data images in folder 'output_spectrum'.")
