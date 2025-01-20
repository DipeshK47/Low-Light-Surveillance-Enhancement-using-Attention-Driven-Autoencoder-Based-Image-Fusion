import numpy as np
import torch
import os
from PIL import Image
from utils_didfuse import Test_fusion

# Test Details 

device = "cpu"

addition_mode = 'Sum'  # 'Sum' & 'Average' & 'l1_norm'

test_data_path = '/Users/dipeshkumar/Downloads/IVIF-DIDFuse 2/Datasets/LLVI'  # Change testing data path

# Determine the number of files

Test_Image_Number = len(os.listdir(test_data_path))

# Test

for i in range(int(Test_Image_Number / 2)):
    Test_IR = Image.open(test_data_path + '/IR' + str(i + 1) + '.jpg')  # infrared image
    Test_Vis = Image.open(test_data_path + '/VIS' + str(i + 1) + '.jpg')  # visible image

    # Fusion logic
    Fusion_image = Test_fusion(Test_IR, Test_Vis)
    
    # Normalize the data to the [0, 255] range
    Fusion_image_normalized = np.interp(Fusion_image, (Fusion_image.min(), Fusion_image.max()), (0, 255))
    Fusion_image_normalized = Fusion_image_normalized.astype(np.uint8)

    # Convert to Pillow Image and save
    Fusion_image_pil = Image.fromarray(Fusion_image_normalized)
    Fusion_image_pil = Fusion_image_pil.convert("L")  # convert to grayscale if needed
    Fusion_image_pil.save('/Users/dipeshkumar/Downloads/IVIF-DIDFuse 2/Test_result/new' + str(i + 1) + '.jpg')  # change image save path
