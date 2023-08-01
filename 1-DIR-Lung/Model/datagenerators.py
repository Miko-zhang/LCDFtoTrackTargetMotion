import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data


class Dataset(Data.Dataset):
    def __init__(self, train_dir,vol_size):
        # 初始化
        self.vol_size = vol_size
        self.fixed_files = []
        self.moving_files = []
        
        patient_dirs = os.listdir(train_dir)
        for p_dir in patient_dirs:
            moving_files = []
            fixed_files = []
            
            path = os.path.join(train_dir,p_dir)
            # 10%-90% phase
            moving_files = glob.glob(os.path.join(path,'[10-90]*.nii.gz'))
            # reference phase
            fixed_files = glob.glob(os.path.join(path,'0.nii.gz'))*len(moving_files)
            
            self.fixed_files += fixed_files
            self.moving_files += moving_files

    def __len__(self):
        return len(self.moving_files)

    def __getitem__(self, index):

        fixed_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.fixed_files[index]))[np.newaxis, ...]
        moving_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.moving_files[index]))[np.newaxis, ...]
        
        fixed_img_arr = self.__centercrop__(fixed_img_arr,self.vol_size)
        moving_img_arr = self.__centercrop__(moving_img_arr,self.vol_size)
        
        return (fixed_img_arr,moving_img_arr)
    
    def __centercrop__(self,arr,vol_size):
        
        _,D,W,H = arr.shape
        
        w0 = int((W - vol_size[1])/2)
        w1 = w0 + vol_size[1]
        
        h0 = int((H - vol_size[2])/2)
        h1 = h0 + vol_size[2]
        
        
        if D > vol_size[0]:
            d0 = int((D - vol_size[0])/2)
            d1 = d0 + vol_size[0]
            new_arr = arr[:,d0:d1,w0:w1,h0:h1]
            
        else:# D <= vol_size[0]
            new_arr = arr[:,:,w0:w1,h0:h1]
            tap0 = int((vol_size[0] - D)/2)
            tap1 = vol_size[0] - D - tap0
            new_arr = np.pad(new_arr,((0,0),(tap0,tap1),(0,0),(0,0)),'constant',constant_values=(0))
            
        return new_arr
