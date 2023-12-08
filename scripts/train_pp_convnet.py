import sys
import os
import argparse
from ..src.image_segmentation import ImageSegmentation




def parse_args():
    parser = argparse.ArgumentParser(description='Post process training script')
    parser.add_argument('--unet-model', type=str, help='Model with which to initialize UNet.')
    parser.add_argument('--preprocess', type=str, required='--unet-model' not in sys.argv, help='Preprocessing mode (v1 or v2) -- both produce a 6-channel feature image, the features differ slightly, see ClassMethod generate_feature_image for more details.')
    parser.add_argument('--data-path', type=str, required=True, help='Absolute path to the model output directory.')
    parser.add_argument('--feature-images', type=str, required=True, help='Absolute path to training data HDF5.')
    parser.add_argument('--logit-images', type=str, required=True, help='Absolute path to training data HDF5.')
    parser.add_argument('--mask-images', type=str, required=True, help='Absolute path to training data HDF5.')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of patches per batch')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--spatial-attention', action='store_true', help='Use Spatial Attention Module on Embedding')
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.data_path, exist_ok=True)
    
    image_segmentation = ImageSegmentation(model_weights_path=args.unet_model,
                                           lr=args.lr,spatial_attention=args.spatial_attention,
                                           post_process=True)

    image_segmentation.train_pp_convnet_model(args.data_path, args.feature_images, 
                                   args.logit_images,args.mask_images, args.batch_size, 
                                   args.epochs)
    
if __name__ == "__main__":
    main()