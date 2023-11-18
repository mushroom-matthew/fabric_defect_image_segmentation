import sys
import os
import argparse
import h5py
from ..src.image_segmentation import ImageSegmentation
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation Assessment Script')
    parser.add_argument('--skip-extract', action='store_true', help='Skip patch prep (inital extraction of patches from images)')
    parser.add_argument('--skip-refine', action='store_true', help='Skip the data refinement, i.e. even out class sampling')
    parser.add_argument('--preprocess', type=str, required=True, help='Preprocessing mode (v1 or v2) -- both produce a 6-channel feature image, the features differ slightly, see ClassMethod generate_feature_image for more details.')
    parser.add_argument('--image-dirs', nargs='+', type=str, required=True, help='Absolute path(s) to the input directory of images')
    parser.add_argument('--mask-dirs', nargs='+', type=str, help='Absolute path(s) to the input directory of masks.')
    parser.add_argument('--grain', type=int, default=64, help='Stride of patch extraction (smaller grain --> more patches --> more accurate --> longer runtime)')
    parser.add_argument('--training-testing-split', type=float, default=0.83333, help='Decimal equivalent percentage of train/test split (fyi, validation data will be split later from train data).')
    parser.add_argument('--max-clean-tt-patches-per-image', type=int, default=25, help='Each image will produce a minimum of 256 possible training patches (grain=64). Using all data from the nominal dataset at this grain, ~56K training patches (mostly defect free) may be generated. This parameter can help to manage training data sizes while preferencing data from known defects.')
    parser.add_argument('--training-folder', type=str, required=True, help='Absolute path to the training data folder')
    parser.add_argument('--training-validation-split', type=float, default=0.83333, help='Decimal equivalent percentage of train/validation split.')
    parser.add_argument('--max-tv-patches', type=int, default=5000, help='Size of training+validation samples')
    return parser.parse_args()

def main():
    args = parse_args()
    assert args.grain <= 64, 'Please do not choose a grain larger than the network input patch size (64x64)'
    assert args.grain >= 16, 'Please do not choose a grain smaller than 16 as it may cause memory issues during reconstruction.'
    
    os.makedirs(args.training_folder, exist_ok=True)
    
    image_segmentation = ImageSegmentation(None,mode=args.preprocess)

    pretrain_outfile = os.path.join(args.training_folder,f'pretraining_patches_p{args.preprocess}.h5')
    test_outfile = os.path.join(args.training_folder,f'testing_patches_p{args.preprocess}.h5')

    if not args.skip_extract:

        move_existing_training_data(pretrain_outfile)
        move_existing_training_data(test_outfile)           

        image_paths = []
        for directory in args.image_dirs:
            if os.path.isdir(directory):
                for root, dirs, files in os.walk(directory):
                    image_paths.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])
            else:
                raise Exception('Input image directory was expected, but received something else.')

        mask_paths = []
        for directory in args.mask_dirs:
            if os.path.isdir(directory):
                for root, dirs, files in os.walk(directory):
                    mask_paths.extend([os.path.join(root, file) for file in files if file.lower().endswith('.png')])
            else:
                raise Exception('Input mask directory was expected, but received something else.')

        training_patches = []
        testing_patches = []
        #print(image_paths)
        for image_path in image_paths:
            smask_paths = [p for p in mask_paths if os.path.basename(image_path)[:-4] in p]
            print(image_path)
            if len(smask_paths) < 1 and not 'NODefect' in image_path:
                print(f'Skipped {image_path} because there was not a mask but the image is labelled a defect.')
                continue
            
            if len(smask_paths) > 0:
                print(smask_paths)
            else:
                print('Defect free mask')

            defect_patches, clean_patches = image_segmentation.training_patch_extraction(image_path,smask_paths,args.grain,args.max_clean_tt_patches_per_image)
            if defect_patches is None or clean_patches is None:
                ll = '+'.join(smask_paths)
                print(f'Skipped {ll} because loaded mask was blank.')
                continue

            a, b = split_patches_to_training_testing(defect_patches,args.training_testing_split)
            training_patches.extend(a)
            testing_patches.extend(b)
            del a, b

            c, d = split_patches_to_training_testing(clean_patches,args.training_testing_split)
            training_patches.extend(c)
            testing_patches.extend(d)
            del c, d

            training_patches = np.array(training_patches)
            testing_patches = np.array(testing_patches)

            
            save_chunk_h5(pretrain_outfile,training_patches)
            save_chunk_h5(test_outfile,testing_patches)

            del training_patches, testing_patches
            training_patches = []
            testing_patches = []
    
    if not args.skip_refine:
        train_outfile = os.path.join(args.training_folder,f'training_patches_p{args.preprocess}.h5')
        validation_outfile = os.path.join(args.training_folder,f'validation_patches_p{args.preprocess}.h5')

        #move_existing_training_data(train_outfile)
        #move_existing_training_data(validation_outfile)

        N_samples = args.max_tv_patches
        N_train = int(N_samples*args.training_validation_split)
        N_val = N_samples-N_train
        print(f'{N_train} training samples, {N_val} validation samples')
        with h5py.File(pretrain_outfile,'r') as training:
            training_i = np.arange(training['patches'].shape[0])
            print(f'Quantifying classes in training/validation.')
            n_class = np.zeros((len(image_segmentation.defect_labels),))
            any_class = np.zeros((len(training_i),))
            for i in training_i:
                if np.mod(i,100) == 0:
                    print(f'\rAssessing patch {i} of {len(training_i)}',end='',flush=True)
                any_class[i] = np.amax(training['patches'][i][:,:,6:],axis=(0,1))
                n_class[int(any_class[i])] += 1
            print('')
            print(n_class)
            minClass=np.min(n_class)
            
            if minClass*len(image_segmentation.defect_labels) > args.max_tv_patches:
                minClass = int(args.max_tv_patches/len(image_segmentation.defect_labels))
            
            print(f'Refining training/validation set to have a minimum of {minClass} samples per class.')
            min_N_train = int(minClass*args.training_validation_split)
            min_N_val = minClass - min_N_train 

            n_class_train = np.zeros((len(image_segmentation.defect_labels),))
            n_class_val = np.zeros((len(image_segmentation.defect_labels),))
            n_train = 0
            n_val = 0

            np.random.shuffle(training_i)            
            while n_train < N_train or n_val < N_val:
                for i,r in enumerate(training_i):
                    m = int(any_class[r])
                    if n_class_train[m] < min_N_train:
                        print(f'Placing patch {r} in training class {image_segmentation.defect_labels[m]} : {n_train+1} / {N_train}')
                        save_chunk_h5(train_outfile,training['patches'][r])
                        n_train += 1
                        n_class_train[m] += 1
                    elif n_class_val[m] < min_N_val:
                        print(f'Placing patch {r} in validation class {image_segmentation.defect_labels[m]}: {n_val+1} / {N_val}')
                        save_chunk_h5(validation_outfile,training['patches'][r])
                        n_val += 1
                        n_class_val[m] += 1
                    elif np.all(np.greater_equal(n_class_train,min_N_train)) and n_train < N_train:
                        print(f'Placing patch {r} in training class {image_segmentation.defect_labels[m]} : {n_train+1} / {N_train}')
                        save_chunk_h5(train_outfile,training['patches'][r])
                        n_train += 1
                        n_class_train[m] += 1
                    elif np.all(np.greater_equal(n_class_val,min_N_val)) and n_val < N_val:
                        print(f'Placing patch {r} in validation class {image_segmentation.defect_labels[m]} : {n_val+1} / {N_val}')
                        save_chunk_h5(validation_outfile,training['patches'][r])
                        n_val += 1
                        n_class_val[m] += 1

def move_existing_training_data(file_path):
    if os.path.isfile(file_path):
        # If it exists, find the next available version
        version = 1
        while os.path.isfile(f"{file_path}_v{version:02d}.h5"):
            version += 1

        # Rename the existing file with the incremental version tag
        os.rename(file_path, f"{file_path}_v{version:02d}.h5")

def save_chunk_h5(file_path, data):
    # Check if the file already exist

    with h5py.File(file_path, 'a') as file:
        if 'patches' not in file:
            dataset = file.create_dataset('patches', shape=(0, 64, 64, 1), maxshape=(None, 64, 64, 19), dtype='f')
        else:
            dataset = file['patches']

        # Append the data
        current_size = dataset.shape[0]
        if len(data.shape) < 4:
            A = 1
        else:
            A = data.shape[0]
        
        new_size = current_size + A
        dataset.resize((new_size, 64, 64, data.shape[-1]))
        dataset[current_size:new_size] = data       

def split_patches_to_training_testing(patches,split):
    training_num = len(patches)*split
    #print(training_num)
    training_patches = []
    testing_patches = []
    patch_indices = np.arange(len(patches))
    np.random.shuffle(patch_indices)
    count = 0
    for i in patch_indices:
        if count < np.ceil(training_num):
            training_patches.append(patches[i])
        else:
            testing_patches.append(patches[i])
        count += 1

    return training_patches, testing_patches
        
if __name__ == "__main__":
    main()  