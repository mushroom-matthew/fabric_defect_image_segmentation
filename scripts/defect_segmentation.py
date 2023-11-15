import sys
import os
import argparse
import matplotlib.pyplot as plt
from ..src.image_segmentation import ImageSegmentation
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Defect Segmentation Script')
    parser.add_argument('--model', type=str, required=True, help='Absolute path to the model weights file')
    parser.add_argument('--image', type=str, required=True, help='Absolute path to the input image file or directory')
    parser.add_argument('--grain', type=int, default=64, help='Stride of patch extraction (smaller grain --> more patches --> more accurate --> longer runtime)')
    parser.add_argument('--post-process', type=str, default='argmax', help='Post-processing method (argmax or prob_thresh)')
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

    # Get a list of input image paths if a directory is provided
    if os.path.isdir(args.image):
        image_paths = [os.path.join(args.image, file) for file in os.listdir(args.image) if file.lower().endswith('.png')]
    else:
        image_paths = [args.image]

    # Create results folder if it doesn't exist
    if args.results_folder:
        os.makedirs(args.results_folder, exist_ok=True)

    # Create reports folder if it doesn't exist
    if args.reports_folder:
        os.makedirs(args.reports_folder, exist_ok=True)

    # Process each input image
    for image_path in image_paths:
        segmented_image, likely_seg_image = image_segmentation.segment_image(image_path, args.grain, args.post_process)
        
        report_df = image_segmentation.analyze_segmented_mask(segmented_image)
        print(image_path)
        print('------------------  Defect Report  ------------------')
        print(report_df.to_string(index=False))
        print('-----------------------------------------------------')
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
            output_path = os.path.join(args.reports_folder, f'{image_name}_defect_report_{image_segmentation.mode}.csv')
            report_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
