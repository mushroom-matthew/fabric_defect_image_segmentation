import os
import numpy as np
import skimage as ski
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import mode
import pandas as pd
from scipy.ndimage import uniform_filter
import h5py

def recall(predicted, ground_truth):
    true_positive = np.sum(np.logical_and(ground_truth, predicted))
    false_negative = np.sum(np.logical_and(ground_truth, np.logical_not(predicted)))
    if true_positive + false_negative == 0:
        return 0
    
    recall = true_positive / (true_positive + false_negative)
    return recall

def precision(predicted, ground_truth):
    true_positive = np.sum(np.logical_and(ground_truth, predicted))
    false_positive = np.sum(np.logical_and(np.logical_not(ground_truth), predicted))
    precision = true_positive / (true_positive + false_positive)
    return precision

def dice(predicted, ground_truth):
    intersection = np.logical_and(ground_truth, predicted)
    dice = (2 * np.sum(intersection)) / (np.sum(ground_truth) + np.sum(predicted))
    return dice

def jaccard(predicted, ground_truth):
    intersection = np.logical_and(ground_truth, predicted)
    union = np.logical_or(ground_truth, predicted)
    
    # Avoid division by zero
    if np.sum(union) == 0:
        return 0.0
    
    return np.sum(intersection) / np.sum(union)

def probability_thresholding_default_zero(probabilities, threshold):
    # Assuming probabilities is a 2D array where each row represents a pixel's probabilities for different classes
    class_labels = np.argmax(probabilities, axis=-1)  # Select the class with the highest probability
    max_probabilities = probabilities.max(axis=-1)

    # Apply threshold to probabilities, defaulting to 0 if no class passes the threshold
    mask = np.where(max_probabilities > threshold, class_labels, 0)

    return mask

def accuracy(predicted, ground_truth):
    # Ensure both arrays have the same shape
    assert ground_truth.shape == predicted.shape, "Arrays must have the same shape"

    # Calculate accuracy
    correct_predictions = np.sum(ground_truth == predicted)
    total_predictions = np.prod(ground_truth.shape)

    accuracy_value = correct_predictions / total_predictions
    return accuracy_value


def illumination_normalization(image,beta): # adapted from https://www.hindawi.com/journals/isrn/2013/516052/
    gmage = np.log(image.astype(np.float32))

    ghat = uniform_filter(gmage,size=7,mode='mirror')

    d = gmage - ghat

    alpha = np.mean(np.abs(d))

    h = np.exp(np.divide(d,alpha*beta))
    o = np.minimum(h,1)

    return o

def reconstruct_image_from_patches(patches, stride, original_shape, agg_mode='mode'):
    # Calculate the dimensions of the reconstructed image
    num_rows, num_cols, num_chan = original_shape
    patch_height, patch_width, patch_depth = patches.shape[1], patches.shape[2], patches.shape[3]
    stride_height, stride_width = map(int,stride)

    # Calculate the maximum number of times a pixel was sampled
    max_samples_height = np.ceil(patch_height / stride_height).astype(np.int16)
    max_samples_width = np.ceil(patch_width / stride_width).astype(np.int16)

    # Initialize a 3D matrix with NaNs
    reconstructed_image = np.empty((num_rows, num_cols, num_chan, max_samples_height * max_samples_width))
    reconstructed_image[:] = np.nan
    #print(reconstructed_image.shape)
    #print(patch_height,patch_width,patch_depth)
    vw = ski.util.view_as_windows(reconstructed_image[:,:,:,0],
                                  (patch_height, patch_width, patch_depth),
                                  step=(stride_height,stride_width,patch_depth))
    # Iterate through patches and populate the 3D matrix
    idx = 0
    for i in range(patches.shape[0]):
        print(f"\rReconstructing with patch {i} of {patches.shape[0]}",end='',flush=True)
        col = (i % vw.shape[1]) * stride_width
        row = (i // vw.shape[1]) * stride_height
        
        reconstructed_image[row:row + patch_height, col:col + patch_width, :, idx] = patches[i]
        
        idx += 1
        if idx == max_samples_height * max_samples_width:
            idx = 0
    print(f"\n")

    # Calculate the mode along the third dimension
    if agg_mode == 'mode':
        reconstructed_image = mode(reconstructed_image, axis=3, nan_policy='omit')[0]
    elif agg_mode == 'mean':
        reconstructed_image = np.nanmean(reconstructed_image, axis=3)

    return np.nan_to_num(reconstructed_image, nan=1)

class ImageSegmentation:
    def __init__(self, model_weights_path, mode=None):
        # Load your pretrained UNet model
        self.defect_labels = ['000','002','006','010',
                              '016','019','022','023',
                              '025','027','029','030','036']
        
        self.unet = self.initialize_unet()

        if not mode and not model_weights_path:
            Exception('Please input either a model or a preprocessing mode')
        
        if model_weights_path:
            if 'v2' in model_weights_path:
                self.mode = 'v2'
            else:
                self.mode = 'v1'

            self.load_pretrained_model(model_weights_path)
        else:
            self.mode=mode

        # Sizes of the model input and output
        self.input_size = self.unet.input_shape[1:]
        self.output_size = self.unet.output_shape[1:]
        self.threshold = 0.5

    def initialize_unet(self):
        # Define and load your UNet model architecture
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(64, 64, 6),
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(64, 64, 6),
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),  # Add Dropout
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform'),
        ])

        x = encoder.output
        x = tf.keras.layers.Conv2DTranspose(32,(3,3),strides=(2,2), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Concatenate()([x,encoder.layers[6].output])
        x = tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(1,1), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2DTranspose(64,(3,3),strides=(2,2), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Concatenate()([x,encoder.layers[3].output])
        x = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(1,1), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2DTranspose(128,(3,3),strides=(2,2), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64,(5,5),strides=(1,1), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(64,(5,5),strides=(1,1), activation='relu', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Conv2D(len(self.defect_labels),(1,1),strides=(1,1), activation='softmax', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        
        # Load the weights
        unet = tf.keras.models.Model(inputs=encoder.input, outputs=x)
        unet.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalFocalCrossentropy(),
                     metrics=[tf.keras.metrics.CategoricalAccuracy(),
                              tf.keras.metrics.CategoricalCrossentropy()])
        
        return unet

    def load_pretrained_model(self, model_weights_path):
        
        self.unet.load_weights(model_weights_path)

        return

    def load_image(self, image_path):
        # Load an image from the given path
        image = ski.io.imread(image_path)
        if len(image.shape) > 2:
            image = image[:, :, 0]

        if image.shape[0] < 256:
            image = np.pad(image,((256-image.shape[0],0),(0,0)),mode='constant',constant_values=255)

        if image.shape[1] < 4096:
            image = np.pad(image,((0,0),(4096-image.shape[1],0)),mode='constant',constant_values=255)

        if image.shape[0] > 256 or image.shape[1] > 4096:
            raise Exception("Loaded image is larger in at least one of the image dimensions.")

        return image

    def watershed_image(self, image):
        image = ski.filters.gaussian(image,13)
        image = ski.filters.sobel(image)

        peak = np.sum(image,0).argmax()

        steps = [-50,50]
        marker_start = np.zeros_like(image)
        for j,step in enumerate(steps):
            marker_start[:,np.min((np.max((0,peak+step)),image.shape[1]))] = j+1

        segmented = ski.segmentation.watershed(image,marker_start)
        return segmented
       
    def generate_feature_images(self, image_path):
        # Load and preprocess a single image
        image = self.load_image(image_path)
        
        image11 = ski.exposure.rescale_intensity(1.0*image,in_range=(0,255))
        image12 = ski.util.invert(image11)
        if self.mode == 'v1':
            image21 = ski.exposure.equalize_adapthist(image11,16)
            image22 = ski.exposure.equalize_adapthist(image12,16)
            image31 = ski.exposure.adjust_sigmoid(image21)
            image32 = ski.exposure.adjust_sigmoid(image22)
        elif self.mode == 'v2':
            image21 = ski.exposure.equalize_adapthist(image11,16)
            image22 = ski.exposure.equalize_adapthist(image12,16)
            image21 = ski.exposure.adjust_sigmoid(image21)
            image22 = ski.exposure.adjust_sigmoid(image22)
            image31 = illumination_normalization(image,0.5)
            image32 = ski.util.invert(image31)
        
        segmented = self.watershed_image(image11)
        background_mask = np.equal(segmented,1).astype(np.uint8)

        feature_image = np.concatenate((image11[:,:,np.newaxis],
                                    image21[:,:,np.newaxis],
                                    image31[:,:,np.newaxis],
                                    image12[:,:,np.newaxis],
                                    image22[:,:,np.newaxis],
                                    image32[:,:,np.newaxis]),axis=2)
        
        return feature_image, background_mask

    def split_to_patches(self,image,patch_size,step):
        patches = ski.util.view_as_windows(image, patch_size, step=step)
        if len(patch_size) == 3:
            patches = patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])
        elif len(patch_size) == 2:
            patches = patches.reshape(-1, patch_size[0], patch_size[1])

        return patches

    def segment_image(self, image_path, grain=8, post_process='argmax'):
        # Split input image into foreground, background and generate feature image
        feature_image, background_mask = self.generate_feature_images(image_path)

        feature_patches = self.split_to_patches(feature_image,self.input_size,(grain,grain,self.input_size[2]))
        background_patches = self.split_to_patches(background_mask,self.input_size[0:2],(grain,grain))
        
        p = []
        for i,(f,b) in enumerate(zip(feature_patches,background_patches)):
            #print(f.shape)
            #print(b.shape)
            print(f'Patch {i} of {len(feature_patches)}')
            p.append(np.zeros(self.output_size))  
            if np.any(b):
                p[-1][:, :, 0][b==1] = 1  # Update only the 0-index where b is True to 1
                p[-1][:, :, 0][b==0] = np.nan
                #print(p[-1][:,:,0])
                print('Background patch')
            else:
                # Perform segmentation using the loaded model
                o = self.unet.predict(np.expand_dims(f, axis=0))
                p[-1] = o.squeeze()  # Use squeeze to remove dimensions of size 1

        likely_seg_image = reconstruct_image_from_patches(np.array(p), (grain, grain), 
                                                          (256, 4096, 13), agg_mode='mean')
        

        if post_process == 'argmax':
            segmented_image = np.argmax(likely_seg_image,axis=-1)
        elif post_process == 'prob_thresh':
            segmented_image = probability_thresholding_default_zero(likely_seg_image,threshold=self.threshold)
        else:
            Exception('Please provide a supported post processing type')
        #print(f'{segmented_image.shape}')

        return segmented_image, likely_seg_image

    def analyze_segmented_mask(self, segmented_image):

        regions = ski.measure.regionprops(segmented_image)

        # Initialize lists to store properties
        labels = []
        areas = []
        eccentricities = []
        loc_row = []
        loc_col = []

        # Iterate through each labeled region
        for region in regions:
            # Extract properties of connected components within the labeled region
            connected_components = ski.measure.label(region.image)

            # Iterate through each connected component
            for component_label in np.unique(connected_components)[1:]:
                component_mask = connected_components == component_label
                component_props = ski.measure.regionprops(component_mask.astype(np.uint8))

                # Extract properties
                label = self.defect_labels[region.label]
                area = component_props[0].area
                eccentricity = component_props[0].eccentricity
                centroid = component_props[0].centroid

                # Append properties to lists
                labels.append(label)
                areas.append(area)
                eccentricities.append(eccentricity)
                loc_row.append(centroid[0])
                loc_col.append(centroid[1])

        # Create a DataFrame with the collected properties
        data = {
            'Label': labels,
            'Area': areas,
            'Eccentricity': eccentricities,
            'Location_row': loc_row,
            'Location_col': loc_col
        }

        df = pd.DataFrame(data)

        return df

    def prepare_mask(self, image, mask_paths=None, type='label'):
        mask_image = np.zeros_like(image,dtype=np.float32)
        for mask_path in mask_paths:
            loi = [i for i,label in enumerate(self.defect_labels) if f"_{label}_" in mask_path]
            imask_image = self.load_image(mask_path)
            mask_image += float(loi[0])*(imask_image>0).astype(np.float32)
            del imask_image
            #print(np.unique(mask_image))
            if len(np.unique(mask_image)) < 2:
                return None

        mask_image = mask_image.astype(np.int16)
        if type == 'one-hot':
            mask_image = np.eye(len(self.defect_labels))[mask_image]

        return mask_image

    def evaluate_performance(self, image_path, mask_paths, grain, post_process):
        # Evaluate the model's performance against a given mask
        segmented_image, likely_seg_image = self.segment_image(image_path,grain=grain,post_process=post_process)
        
        mask_image = self.prepare_mask(segmented_image,mask_paths,type='label')
        
        labs = np.union1d(np.unique(mask_image),np.unique(segmented_image))
        def_l = [self.defect_labels[int(lab)] for lab in labs]
        acc = []
        prec = []
        rec = []
        dice_ = []
        iou = []

        for lab in labs:
            S = (segmented_image == lab).astype(np.uint8)
            M = (mask_image == lab).astype(np.uint8)
            a,i,d,p,r = self.compute_performance_metrics(S,M)

            acc.append(a)
            prec.append(p)
            rec.append(r)
            dice_.append(d)
            iou.append(i)

        performance_metrics = pd.DataFrame({'Label':def_l,
                                            'Accuracy':acc,
                                            'Precision': prec,
                                            'Recall': rec,
                                            'Dice': dice_,
                                            'IoU':iou})

        return performance_metrics, segmented_image, likely_seg_image

    def compute_performance_metrics(self, segmented_image, mask_image):
        acc = accuracy(segmented_image, mask_image)
        iou = jaccard(segmented_image, mask_image)
        dice_ = dice(segmented_image, mask_image)
        prec = precision(segmented_image, mask_image)
        rec = recall(segmented_image, mask_image)
        return acc, iou, dice_, prec, rec


    def training_patch_extraction(self, image_path, mask_paths, grain, max_clean):
        feature_image, background_mask = self.generate_feature_images(image_path)

        mask_image = self.prepare_mask(feature_image[:,:,0],mask_paths)
        if mask_image is None:
            return None, None
        #print(mask_image.shape)
        feature_patches = self.split_to_patches(feature_image,self.input_size,(grain,grain,self.input_size[2]))
        background_patches = self.split_to_patches(background_mask,self.input_size[0:2],(grain,grain))
        mask_patches = self.split_to_patches(mask_image,self.output_size[0:2],(grain,grain))


        del feature_image, background_mask, mask_image
        patch_indices = np.arange(feature_patches.shape[0])
        np.random.shuffle(patch_indices)

        d = []
        p = []
        clean_patches_from_this_image = 0
        for i in patch_indices:
            #print(f.shape)
            #print(b.shape)
            #print(f'Patch {i} of {len(feature_patches)}')
            # p.append(np.zeros(self.output_size))  
            if np.any(background_patches[i]):
                #print('Ommitting background patch')
                continue
                #print(p[-1][:,:,0])
            elif np.any(mask_patches[i]>0):
                d.append(np.concatenate((feature_patches[i],mask_patches[i][:,:,np.newaxis]),axis=-1))
            else:
                if clean_patches_from_this_image < max_clean:
                    p.append(np.concatenate((feature_patches[i],mask_patches[i][:,:,np.newaxis]),axis=-1))
                    clean_patches_from_this_image += 1

        return d,p

    def train_model(self, data_path, training_data, validation_data, batch_size=32, epochs=10):
        if not training_data or not validation_data:
            raise ValueError("Training or validation data is not provided.")

        training = h5py.File(training_data)
        #print(training['patches'].shape)
        validation = h5py.File(validation_data)
        # Define the ImageDataGenerator for training data
        train_datagen = ImageDataGenerator(
            # No rescaling needed
            # Randomly flip images horizontally
            horizontal_flip=True,
            # Randomly flip images vertically
            vertical_flip=True,
            # Convert y to one-hot encoding
            #preprocessing_function=self.one_hot_encoding
        )

        # Define the ImageDataGenerator for validation data
        val_datagen = ImageDataGenerator(
            # No rescaling needed for validation data
            # Convert y to one-hot encoding
            #preprocessing_function=self.one_hot_encoding
        )

        # Define the generators for training and validation data
        train_generator = train_datagen.flow(
            x=training['patches'][:,:,:,0:6],
            y=self.one_hot_encoding(training['patches'][:,:,:,6]),
            batch_size=batch_size,
            shuffle=True
        )

        val_generator = val_datagen.flow(
            x=validation['patches'][:,:,:,0:6],
            y=self.one_hot_encoding(validation['patches'][:,:,:,6]),
            batch_size=batch_size,
            shuffle=False  # No need to shuffle validation data
        )

        # Define the path for saving weights after each epoch
        intermediate_dir = f'{data_path}/intermediate_weights'
        os.makedirs(intermediate_dir, exist_ok=True)

        # Define the ModelCheckpoint callback to save weights after each epoch
        checkpoint_filepath = f'{intermediate_dir}/{self.mode}_weights_epoch_{{epoch:02d}}.h5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            verbose=1
        )

        # Train the model
        history = self.unet.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[model_checkpoint_callback]
        )
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'{data_path}/{self.mode}_training_history.csv', index=False)
        # Save the final model
        self.unet.save_weights(f'{data_path}/final_model_weights_{self.mode}.h5')

    def one_hot_encoding(self, y):
        # Convert y to one-hot encoding
        one_hot_y = tf.one_hot(y, depth=len(self.defect_labels), axis=-1)
        return one_hot_y
    # def select_optimal_threshold(self, thresholds, metric='iou'):
    #     best_metric_value = 0
    #     best_threshold = 0

    #     for threshold in thresholds:
    #         predicted_masks = self.apply_thresholds(self.generate_probability_scores(), threshold)
    #         iou, _, _, _, _ = self.evaluate_performance(predicted_masks)

    #         if metric == 'iou' and iou > best_metric_value:
    #             best_metric_value = iou
    #             best_threshold = threshold

    #     return best_threshold