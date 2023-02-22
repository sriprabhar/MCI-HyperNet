import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factors,dataset_types,mask_types,train_or_valid,modeltaskA, current_dataset_type, device): 
        # List the h5 files in root 
        #files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.datasetdict = {'mrbrain_t1':1,'mrbrain_flair':2,'mrbrain_ir':3}
        self.maskdict={'cartesian':1,'gaussian':2}
        self.modeltaskA = modeltaskA
        self.current_dataset_type = current_dataset_type
        self.device = device
        for dataset_type in dataset_types:
            dataroot = os.path.join(root, dataset_type)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot, mask_type,train_or_valid)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor))).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            #acc_factor = float(acc_factor[:-1].replace("_","."))
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice,acc_factor,mask_type,dataset_type = self.examples[i] 
    
        if dataset_type == self.current_dataset_type[0]: # learning new contexts - if the context is a new one to be  currently learned then use its target for learning
            with h5py.File(fname, 'r') as data:
               
                acc_val = float(acc_factor[:-1].replace("_","."))
                currentdatatypetarget = data['volfs'][:,:,slice].astype(np.float64)# converting to double
                key_img = 'img_volus_{}'.format(acc_factor)
                key_kspace = 'kspace_volus_{}'.format(acc_factor)

                input_img  = data[key_img][:,:,slice]
            #print(key_img)
                input_kspace  = data[key_kspace][:,:,slice]

                input_kspace = npComplexToTorch(input_kspace)
                mask_val = self.maskdict[mask_type] 
                dataset_val = self.datasetdict[dataset_type]
                gamma_input_final = np.array([acc_val, mask_val,dataset_val])
                target = currentdatatypetarget
                #print("new")
                #print("if loop data type:", dataset_type)
        else: # Preserving old contexts - if the acquisiton context is an old one, then its target is not available, use the model trained on the old contexts (modeltaskA) and get the pseudo targets for preserve the knowledge about the context
            #print(type(fname),type(dataset_type), type(self.current_dataset_type[0]))
            newfnamestr = str(fname).replace(str(dataset_type),str(self.current_dataset_type[0]))
            newfname = pathlib.Path(newfnamestr)
            with h5py.File(newfname, 'r') as data:
                acc_val = float(acc_factor[:-1].replace("_","."))
                currentdatatypetarget = data['volfs'][:,:,slice].astype(np.float64)# converting to double
                key_img = 'img_volus_{}'.format(acc_factor)
                key_kspace = 'kspace_volus_{}'.format(acc_factor)
                input_img  = data[key_img][:,:,slice] 
                input_kspace  = data[key_kspace][:,:,slice]
                input_torch = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).to(self.device)
                input_torch = input_torch.float()
                input_kspace = npComplexToTorch(input_kspace)
                input_kspace_torch = input_kspace.unsqueeze(0).unsqueeze(0).to(self.device)
                dataset_val = self.datasetdict[dataset_type]
                mask_val = self.maskdict[mask_type]
                gamma_input_final = np.array([acc_val, mask_val,dataset_val])
                gamma_input_torch = torch.from_numpy(gamma_input_final).unsqueeze(0).to(self.device)
                gamma_input_torch = gamma_input_torch.float()

                target = self.modeltaskA(input_torch,input_kspace_torch, gamma_input_torch, [acc_factor], [mask_type], [dataset_type])
                target = target.squeeze(0).squeeze(0)
                target = target.detach().cpu().numpy().astype(np.float64) # detach from the GPU, the learning graph and  return as a fresh tensor


        return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(gamma_input_final), acc_factor, mask_type,dataset_type

 
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root,acc_factor,dataset_type,mask_path):
    def __init__(self, root,acc_factor,dataset_type,mask_type):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.datasetdict = {'mrbrain_t1':1,'mrbrain_flair':2,'mrbrain_ir':3}
        self.maskdict={'cartesian':1,'gaussian':2}

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type,dataset_type = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]

            # Print statements
            #print (input.shape,target.shape)
            acc_val = float(acc_factor[:-1].replace("_","."))
 
            mask_val = self.maskdict[mask_type] 
            dataset_val = self.datasetdict[dataset_type]

            gamma_input = np.array([acc_val, mask_val,dataset_val])
            
            print("Inside SliceDataDev: ",gamma_input)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),str(fname.name),slice, torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type

class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,dataset_type,mask_type,acc_factor):

        # List the h5 files in root 
        newroot = os.path.join(root, dataset_type,mask_type,'validation','acc_{}'.format(acc_factor))
        files = list(pathlib.Path(newroot).iterdir())
        self.examples = []
        self.datasetdict = {'mrbrain_t1':1,'mrbrain_flair':2,'mrbrain_ir':3}
        self.maskdict={'cartesian':1,'gaussian':2}
        #print(files)
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                #acc_factor = float(acc_factor[:-1].replace("_","."))
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice, acc_factor,mask_type,dataset_type = self.examples[i]
        # Print statements 
        #print (type(fname),slice)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor)
            key_kspace = 'kspace_volus_{}'.format(acc_factor)
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double

            # Print statements
            #print (input.shape,target.shape)
            acc_val = float(acc_factor[:-1].replace("_","."))
 
            mask_val = self.maskdict[mask_type] 
            dataset_val = self.datasetdict[dataset_type]
            gamma_input = np.array([acc_val, mask_val,dataset_val])
            #print("Inside SliceDisplayDataDev: ",gamma_input)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(gamma_input), acc_factor, mask_type, dataset_type
 
