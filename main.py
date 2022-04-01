import numpy as np
# import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pyzbar.pyzbar import decode
from pyzbar import pyzbar
import cv2
import glob
from tqdm import tqdm
import module


print(os.getcwd())
###file = './Amt_2.jpg' <-- only in ipynb
file = r"C:\\Users\\sendr\\Documents\\Proj_Massive_Barcode_Reading\Amt_1.jpg"

image = cv2.imread(file)
height, width = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
t, bimage = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
# kernel = 1/3 * kernel
image_sharp = cv2.filter2D(src=bimage, ddepth=-1, kernel=kernel)
barcodes = pyzbar.decode(image_sharp)

list_barcodeData = []
list_barcodeType = []
list_pixel_position = []

for barcode in barcodes:
    (x, y, w, h) = barcode.rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    list_pixel_position.append(((x, y), (x + w, y), (x, y + h), (x + w, y + h)))

    barcodeData = barcode.data.decode("utf-8")
    list_barcodeData.append(barcodeData)

    barcodeType = barcode.type
    list_barcodeType.append(barcodeType)

    # print (barcodeData+barcodeType)

    text = "{} ({})".format(barcodeData, barcodeType)
    text_pos1 = "{}, {}".format(x, y)
    text_pos2 = "{}, {}".format(x + w, y)
    text_pos3 = "{}, {}".format(x, y + h)
    text_pos4 = "{}, {}".format(x + w, y + h)

    text_size = 1
    text_boldness = 4

    cv2.putText(image, text, (x, y - 43), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_boldness)
    cv2.putText(image, text_pos1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_boldness)
    cv2.putText(image, text_pos2, (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_boldness)
    cv2.putText(image, text_pos3, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_boldness)
    cv2.putText(image, text_pos4, (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), text_boldness)

    # Experiment to get bounding box:
    # cv2.circle(image, (x,y), radius=20, color=(18, 25, 0), thickness= 20) <---only for debugging
    # cv2.circle(image, (x+w,y), radius=20, color=(18, 25, 0), thickness= 20) <---only for debugging

# cv2.imwrite(a, image)
divider = 4
dim = (width // divider, height // divider)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# resized2 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
shelby = modulefinder.Label_A ("Red", "Red", "Redneck", 60)

cv2.imshow('', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))