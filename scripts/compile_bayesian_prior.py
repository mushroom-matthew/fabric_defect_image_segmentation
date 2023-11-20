import sys
import os
import argparse
import matplotlib.pyplot as plt
from ..src.image_segmentation import ImageSegmentation
import pandas as pd
import skimage as ski
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Assessment Script')
    parser.add_argument('--image', type=str, required=True, help='Absolute path to the input image file or directory of images')
    parser.add_argument('--mask', type=str, required=True, help='Absolute path to the input image file or directory of images')
    parser.add_argument('--results-folder', type=str, required=('--save-label-mask' in sys.argv or '--save-logits-map' in sys.argv), help='Absolute path to the results output folder')
    return parser.parse_args()

def main():
    args = parse_args()
    image_segmentation = ImageSegmentation(None, mode='v2')  # Initialize the class here

    if os.path.isdir(args.image):
        image_paths = [os.path.join(args.image, file) for file in os.listdir(args.image) if file.lower().endswith('.png')]
    else:
        image_paths = [args.image]

    if os.path.isdir(args.mask):
        p_mask_paths = [os.path.join(args.mask, file) for file in os.listdir(args.mask) if file.lower().endswith('.png')]
    else:
        p_mask_paths = [args.mask]

    prior_table = image_segmentation.compile_prior_table(image_paths,p_mask_paths)
    
    prior_table.to_csv(os.path.join(args.results_folder, 'defect_morphology_table.csv'), index=False)


if __name__ == "__main__":
    main()   