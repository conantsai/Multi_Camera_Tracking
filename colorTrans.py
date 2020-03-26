import numpy as np
from skimage import io, color
from PIL import Image 
import cv2
import matplotlib.pyplot as plt

def colorTrans(source_path, target_path):
    ## The "Source" image for color transform standards
    s = io.imread(source_path)
    ## The "Target" image to color transform
    t = io.imread(target_path)

    ## Convert RGB triplets to LAB triplets
    Slab = color.rgb2lab(s)
    Tlab = color.rgb2lab(t)

    ## Use the following calculation:
    ## (standard deviation of target’s channel / standard deviation of source’s channel) * (source’s channel - mean of source’s channel) +  mean of target’s channel
    SLABmean = Slab.mean(0).mean(0)
    SLABstd = Slab.std(0).std(0)
    TLABmean = Tlab.mean(0).mean(0)
    TLABstd = Tlab.std(0).std(0)
    height, width, channels = t.shape
    for x in range(0, height):
        for y in range(0, width):
            Tlab[x][y][0] = (SLABstd[0]/TLABstd[0])*(Tlab[x][y][0] - TLABmean[0]) + SLABmean[0]
            Tlab[x][y][1] = (SLABstd[1]/TLABstd[1])*(Tlab[x][y][1] - TLABmean[1]) + SLABmean[1]
            Tlab[x][y][2] = (SLABstd[2]/TLABstd[2])*(Tlab[x][y][2] - TLABmean[2]) + SLABmean[2]

    ## Save the result
    Trgb = color.lab2rgb(Tlab)
    ## Rescale to 0-255 and convert to uint8
    Trgb2 = (255.0 / Trgb.max() * (Trgb - Trgb.min())).astype(np.uint8)
    im = Image.fromarray(Trgb2)
    print(Trgb2)
    # im.save('s2.png')

    # plt.subplot(2,2,1)
    # plt.title('Source')
    # plt.imshow(s)
    # plt.subplot(2,2,2)
    # plt.title('Target Original')
    # plt.imshow(t)
    # plt.subplot(2,2,3)
    # plt.title('Target Transform1')
    # plt.imshow(Trgb)
    # plt.subplot(2,2,4)
    # plt.title('Target Transform2')
    # plt.imshow(Trgb2)
    # plt.show()

    return

if __name__ == "__main__":
    source_path = "code\iLIDS-VID_test\i-LIDS-VID\sequences\cam1\person001\cam1_person001_00317.png" 
    target_path = "code\iLIDS-VID_test\i-LIDS-VID\sequences\cam1\person001\cam1_person001_00318.png"

    colorTrans(source_path, target_path)
    pass



