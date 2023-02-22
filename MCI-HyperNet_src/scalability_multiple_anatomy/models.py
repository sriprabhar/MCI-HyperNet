import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
import torch.nn.functional as F

class DataConsistencyLayer(nn.Module):

    def __init__(self, us_mask_path, device):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask_path = us_mask_path
        self.device = device

    def forward(self,predicted_img,us_kspace,acc_factor, mask_string, dataset_string,us_mask):
        
        us_mask_path = os.path.join(self.us_mask_path,dataset_string,mask_string,'mask_{}.npy'.format(acc_factor))
        us_mask_file = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(self.device)

        #print("us_mask_file shape: ", us_mask_file.shape)
        us_mask = us_mask.unsqueeze(2).unsqueeze(0).to(self.device)

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,us_mask.shape)
        #print(us_mask.dtype)
        updated_kspace1  = us_mask * us_kspace 
        updated_kspace2  = (1 - us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1 + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        update_img_abs = updated_img[:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of a convolution layer followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        )
        
                  
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]

        """
        out = self.layers(input)
        
        return out

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class UnetModelWithPyramidDWPAndDC(nn.Module):

    """
    PyTorch implementation of a U-Net model with pyramid dynamic weight prediction (DWP).
    """


    def __init__(self, args, in_chans, out_chans, chans, num_pool_layers, drop_prob, contextvectorsize):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.relu = nn.ReLU() 

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.weightsize = []

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.dwpfilterbank = nn.ModuleList([nn.Sequential(
            nn.Linear(contextvectorsize, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8,chans*chans*3*3),
        )])

        #self.weightsize.append([chans,in_chans,3,3])
        self.weightsize.append([chans,chans,3,3])
        #print("self.weightsize: ",self.weightsize)

        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            self.dwpfilterbank += [nn.Sequential(nn.Linear(contextvectorsize,8), nn.ReLU(), nn.Linear(8,8), nn.ReLU(), nn.Linear(8,(ch*2)*(ch*2)*3*3))]
            #self.weightsize.append([ch*2,ch,3,3])
            self.weightsize.append([ch*2,ch*2,3,3])
            ch *= 2
            #print("self.weightsize: ",self.weightsize)
##################bottleneck layers #########################3
        #self.conv = ConvBlock(ch, ch, drop_prob)
        self.dwp_latent = nn.Sequential(
            nn.Linear(contextvectorsize, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8,ch*ch*3*3),
        )
        self.latentweightsize = [ch,ch,3,3]
        #print("self.latentweightsize: ", self.latentweightsize)
        self.latentinstancenorm=nn.InstanceNorm2d(ch,affine=True)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            #print("ch: ", ch)
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )
        self.dc = DataConsistencyLayer(args.usmask_path, args.device)

    def forward(self,x, k, gamma_val, acc_string, mask_string, dataset_string,mask):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        batch_size = x.size(0)
        batch_outputs=[]
        for n in range(batch_size):
            stack = []
            output = x[n]
            output = output.unsqueeze(0)
            #print("input size: ", output.size())
            xtemp = output
            filterbank=[]
        # Apply down-sampling layers
            for layer in self.down_sample_layers:
                output = layer(output)
                #print("downsample output size: ", output.size())
                stack.append(output)
                output = F.max_pool2d(output, kernel_size=2)
                #print("downsample output size after maxpool: ", output.size())

            for dwp,wtsize in zip(self.dwpfilterbank,self.weightsize):
                #print("gamma shape: ",gamma_val.shape)
                filters = dwp(gamma_val[n])
                #print("filers size: ", filters.size()," weights size : ",wtsize)
                filters = torch.reshape(filters,wtsize)
                filterbank.append(filters)

        #output = self.conv(output)
            latentfilters = self.dwp_latent(gamma_val[n])
            #print("latent filers size: ", latentfilters.size()," weights size : ",self.latentweightsize)
            latentweights = torch.reshape(latentfilters, self.latentweightsize)
            output = self.relu(self.latentinstancenorm(F.conv2d(output, latentweights, bias=None, stride=1, padding=1)))
            output_latent = output
            #print("latent output size: ", output.size())
        # Apply up-sampling layers
            for layer in self.up_sample_layers:
                output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
                #print("upsample output size: ", output.size())
                encoutput = stack.pop()
                #print("encoutput size: ", encoutput.size())
                encoutfinal = F.conv2d(encoutput,filterbank.pop(),bias=None,stride=1,padding=1)
                output = torch.cat([output, encoutfinal], dim=1)
                #print("output size after cat: ", output.size())
                output = layer(output)
            output = self.conv2(output)
            output=output+xtemp
            #print("output shape: ",output.shape," k[n] shape: ",k[n].shape," mask[n] shape: ",mask[n].shape) 
            output = self.dc(output,k[n],acc_string[n], mask_string[n], dataset_string[n],mask[n])
            batch_outputs.append(output)
        output = torch.cat(batch_outputs,dim=0)
        return output


class DC_CNN(nn.Module):
    
    def __init__(self, args, checkpoint_file, n_ch=1,nc=5):
    #def __init__(self, args, n_ch=1,nc=5):
        
        super(DC_CNN,self).__init__()
        
        cnn_blocks = []
        #dc_blocks = []
        checkpoint = torch.load(checkpoint_file)
        self.nc = nc
        
        for ii in range(self.nc): 
            
            #cnn = UnetModelWithPyramidDWPAndDCRes(args,1,1,32,3,0.0,3)
            cnn = UnetModelWithPyramidDWPAndDC(args,1,1,32,3,0.0,(320*320))
            #cnn.load_state_dict(checkpoint['model']) 
            cnn_blocks.append(cnn)
            
            #dc_blocks.append(DataConsistencyLayer(args.usmask_path, args.device))
        
        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        #self.dc_blocks  = nn.ModuleList(dc_blocks)
        
    def forward(self,x, k, gamma_val, acc_string, mask_string, dataset_string,mask):
        x_cnn = x
        for i in range(self.nc):
            x_cnn = self.cnn_blocks[i](x_cnn, k, gamma_val, acc_string, mask_string, dataset_string,mask)
            #x = x + x_cnn
            #x = self.dc_blocks[i](x,k,acc_string)        
        return x_cnn  


