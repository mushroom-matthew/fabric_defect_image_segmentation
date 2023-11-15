import sys
import os
import argparse
import matplotlib.pyplot as plt
from ..src.image_segmentation import ImageSegmentation
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Defect Segmentation Script')
    parser.add_argument('--model', type=str, required=True, help='Absolute path to the model weights file')
    parser.add_argument('--image', type=str, required=True, help='Absolute path to the input image file or directory of images')
    parser.add_argument('--mask', type=str, required=True, help='Absolute path to the input image file or directory of images')
    parser.add_argument('--grain', type=int, default=64, help='Stride of patch extraction (smaller grain --> more patches --> more accurate --> longer runtime)')
    parser.add_argument('--save-label-mask', action='store_true', help='Save mask images (argmax labeled)')
    parser.add_argument('--save-logits-map', action='store_true', help='Save logit images (softmax layer output)')
    parser.add_argument('--save-reports', action='store_true', help='Save reports')
    parser.add_argument('--results-folder', type=str, required=('--save-label-mask' in sys.argv or '--save-logits-map' in sys.argv), help='Absolute path to the results output folder')
    parser.add_argument('--reports-folder', type=str, required='--save-reports' in sys.argv, help='Absolute path to the reports output folder')
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.grain <= 64, 'Please do not choose a grain larger than the network input patch size (64x64)'
    assert args.grain > 0, 'Please do not choose a grain smaller than 1'
    # Load the model
    model_weights_path = args.model
    image_segmentation = ImageSegmentation(model_weights_path)

    if os.path.isdir(args.image):
        image_paths = [os.path.join(args.image, file) for file in os.listdir(args.image) if file.lower().endswith('.png')]
    else:
        image_paths = [args.image]

    if os.path.isdir(args.mask):
        p_mask_paths = [os.path.join(args.mask, file) for file in os.listdir(args.mask) if file.lower().endswith('.png')]
    else:
        p_mask_paths = [args.mask]

    # Create results folder if it doesn't exist
    if args.results_folder:
        os.makedirs(args.results_folder, exist_ok=True)

    # Create reports folder if it doesn't exist
    if args.reports_folder:
        os.makedirs(args.reports_folder, exist_ok=True)

    # Process each input image
    mask_paths = []
    for image_path in image_paths:
        mask_paths = [p for p in p_mask_paths if os.path.basename(image_path)[:-4] in p]
        if len(mask_paths) == 0:
            print(f'There are no input masks with matching name for {image_path}. Please double check your mask input and adjust accordingly.')
            continue

        performance_metrics, segmented_image, likely_seg_image = image_segmentation.evaluate_performance(image_path,mask_paths,args.grain)

        print(image_path)
        print('VS')
        print(mask_paths)
        print('----------  Segmentation Performance Report  ----------')
        print(performance_metrics.to_string(index=False))
        print('-------------------------------------------------------')

        # Extract the image file name without the extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save the segmented image directly using the image name
        if args.save_logits_map:
            for i in range(11):
                output_path = os.path.join(args.results_folder, f'{image_name}_segmented_{image_segmentation.defect_labels[i]}_{image_segmentation.mode}.png')
                plt.imsave(output_path, likely_seg_image[:,:,i])  # Choose an appropriate colormap

        if args.save_label_mask:
            output_path = os.path.join(args.results_folder, f'{image_name}_segmented_labeled_{image_segmentation.mode}.png')
            plt.imsave(output_path, segmented_image)  # Choose an appropriate colormap

        # Save reports if the flag is set
        if args.save_reports:
            # Add code to save reports here
            output_path = os.path.join(args.reports_folder, f'{image_name}_performance_report_{image_segmentation.mode}.csv')
            performance_metrics.to_csv(output_path, index=False)
            
if __name__ == "__main__":
    main()       