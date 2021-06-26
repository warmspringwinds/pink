import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import skimage.io as io
import os

class OpenEDS(data.Dataset):
    
    def __init__(self,
             dataset_base_folder_path='',
             csv_folds_file=None,
             train=True,
             joint_transform=None,
             fold_number=0,
             test=False):
        
        self.joint_transform = joint_transform

        self.train = train

        self.dataset_base_folder_path = dataset_base_folder_path

        data_df = pd.read_csv(csv_folds_file)
        
        self.test = test
        
        if test:
            
            self.df = data_df[data_df['Split'] == 'Test'].reset_index(drop=True)
            
        else:
            split_train_mask = (data_df['Fold'] != 'Fold{}'.format(fold_number))
            if train:
                self.df = data_df[split_train_mask & (data_df['Split'] == 'Train')].reset_index(drop=True)
            else:
                self.df = data_df[(~split_train_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)


    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):

        image_path = os.path.join(self.dataset_base_folder_path, self.df.loc[index, 'Id'])
        annotation_path = os.path.join(self.dataset_base_folder_path, self.df.loc[index, 'Label'])

        #image = io.imread(image_path)
        image = Image.open(image_path).convert('RGB')
        #rgbimg = img.convert('RGB')
        
        if self.test:
            
            annotation = np.zeros((600, 400), dtype=np.uint8)
            
        else:
            
            annotation = np.load(annotation_path)

        # Images are grayscale, so we copy channels
        #image = np.dstack((image, image, image))

        #image = Image.fromarray(image)
        annotation = Image.fromarray(annotation)

        if self.joint_transform is not None:

            image, annotation = self.joint_transform([image, annotation])

        return image, annotation