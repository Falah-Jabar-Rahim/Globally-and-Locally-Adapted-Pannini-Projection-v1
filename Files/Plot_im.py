import cv2
from matplotlib import pyplot as plt


def im_plt(im):
    plt.axis("off")
    plt.imshow(im)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

