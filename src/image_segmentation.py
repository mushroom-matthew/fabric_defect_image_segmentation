import os
import numpy as np
import skimage as ski
import tensorflow as tf
from scipy.stats import mode
import pandas as pd
from scipy.ndimage import uniform_filter


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
    def __init__(self, model_weights_path):
        # Load your pretrained UNet model
        self.defect_labels = ['000','002','006','010','016',
                              '019','022','023','025','030','036']
        
        if 'v2' in model_weights_path:
            self.mode = 'v2'
        else:
            self.mode = 'v1'

        self.unet = self.load_pretrained_model(model_weights_path)

        # Sizes of the model input and output
        self.input_size = self.unet.input_shape[1:]
        self.output_size = self.unet.output_shape[1:]

    def load_pretrained_model(self, model_weights_path):
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
        x = tf.keras.layers.Conv2D(11,(1,1),strides=(1,1), activation='softmax', padding='same',
                                kernel_initializer='glorot_uniform')(x)
        
        # Load the weights
        unet = tf.keras.models.Model(inputs=encoder.input, outputs=x)
        unet.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalFocalCrossentropy(),
                     metrics=[tf.keras.metrics.CategoricalAccuracy(),
                              tf.keras.metrics.CategoricalCrossentropy()])
        unet.load_weights(model_weights_path)

        return unet

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

    def segment_image(self, image_path, grain=8):
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
                                                          (256, 4096, 11), agg_mode='mean')
        

        segmented_image = np.argmax(likely_seg_image,axis=-1)
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

#    def evaluate_performance(self, image_path, mask_path):
        # Evaluate the model's performance against a given mask
#        segmented_image = self.segment_image(image_path,grain=8)
        
#        mask_image = self.load_image(mask_path)   



        #return performance_metrics

 #   def extract_patches(self, image_paths, mask_paths=None):
        # Extract patches from images and masks (if provided)
        # ...

 #       return image_patches, mask_patches

 #  def retrain_model(self, train_data, val_data, epochs=30):
        # Retrain the UNet model with the given training data
        # ...

 #      return trained_model