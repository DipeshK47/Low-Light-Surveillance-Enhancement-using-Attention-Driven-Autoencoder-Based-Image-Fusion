import numpy as np
import torch
from DIDFuse import AE_Encoder,AE_Decoder
import torch.nn.functional as F

device='cpu'

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2

def Test_fusion(img_test1,img_test2,addition_mode='Sum'):
    AE_Encoder1 = AE_Encoder().to(device)
    AE_Encoder1.load_state_dict(torch.load(
    "/Users/dipeshkumar/Downloads/IVIF-DIDFuse 2/Models/Encoder_weight.pkl",    #change path ENCODER
    map_location=torch.device('cpu'))['weight'])
    
    AE_Decoder1 = AE_Decoder().to(device)
    AE_Decoder1.load_state_dict(torch.load(
    "/Users/dipeshkumar/Downloads/IVIF-DIDFuse 2/Models/Decoder_weight.pkl",    #change path DECODER
    map_location=torch.device('cpu'))['weight'])
    AE_Encoder1.eval()
    AE_Decoder1.eval()
    
    img_test1 = np.array(img_test1, dtype='float32')/255
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    
    img_test2 = np.array(img_test2, dtype='float32')/255
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))
    
    img_test1=img_test1 
    img_test2=img_test2 
    
    with torch.no_grad():
        F_i1,F_i2,F_ib,F_id=AE_Encoder1(img_test1)
        F_v1,F_v2,F_vb,F_vd=AE_Encoder1(img_test2)
        
    if addition_mode=='Sum':      
        F_b=(F_ib+F_vb)
        F_d=(F_id+F_vd)
        F_1=(F_i1+F_v1)
        F_2=(F_i2+F_v2)
         
    with torch.no_grad():
        Out = AE_Decoder1(F_1,F_2,F_b,F_d)
     
    return output_img(Out)
