from fill_depth_colorization import fill_depth_colorization
import os
import cv2
import numpy as np
import time
from multiprocessing import Pool

# directories for inpainting, and raw downloaded data
annotated_directory = "/media/data3/awb/kitti/annotated/train/"
inpainted_directory = "/media/data3/awb/kitti/inpainted/train/"
raw_directory = "/media/data3/awb/kitti/raw/"

def inpaint_image(file):
        directory, image = file
        rgb = cv2.imread(os.path.join(raw_directory,directory[:10],directory,"image_03/data/",image), 1) / 255.0
        sparse_depth = cv2.imread(os.path.join(annotated_directory,directory,"proj_depth/groundtruth/image_03/",image), -1) / 256.0

        inpainted_depth = fill_depth_colorization(rgb, sparse_depth)

        cv2.imwrite(os.path.join(inpainted_directory,directory,"d_image03_"+image), np.uint16(inpainted_depth * 256.0))
        cv2.imwrite(os.path.join(inpainted_directory,directory,"rgb_image03_"+image), rgb * 255.0)

if __name__ == '__main__':
        files = []
        for directory in os.listdir(annotated_directory):
                for image in os.listdir(os.path.join(annotated_directory,directory,"proj_depth/groundtruth/image_03")):
                        files.append((directory, image))

        with Pool(48) as p:
                p.map(inpaint_image, files)