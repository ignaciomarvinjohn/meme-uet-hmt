# blip_captions.py
# Create captions using BLIP-2 model
# The model used here is from HuggingFace

import os
from PIL import Image

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import pandas as pd


#=================================================================
# Setup
#=================================================================
dataset_name = "memotion_num"
root_path = "path_to_the_images" # image folder where the images are located

#--------------------------------------------------------------
# HuggingFace models
processor_name = "Salesforce/blip2-opt-6.7b"
model_name = "Salesforce/blip2-opt-6.7b"
#--------------------------------------------------------------

# file suffix for the save file (filename will be "dataset_name + file_suffix")
file_suffix = "_split_blip2-opt-6.7b.csv"

# create an output folder
os.makedirs(dataset_name, exist_ok=True)


#=================================================================
# Initialize pandas dataframe
#=================================================================
# create a blank pandas dataframe
df = pd.DataFrame(columns = ['Filename', 'Captions'])

# create / overwrite existing file
df.to_csv(os.path.join(dataset_name, dataset_name + file_suffix), index=False)


#=================================================================
# Load model
#=================================================================
# initialize CPU/GPU
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

print("Loading model...")

# load model and visual processors
processor = Blip2Processor.from_pretrained(processor_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

print("Done!")


#=================================================================
# Feedforward the images to the model
#=================================================================
image_count = 0

# for each files in the specified directory
for fname in os.listdir(root_path):

    # compose file path
    file_path = os.path.join(root_path, fname)
        
    try:        
        # open image
        raw_image = Image.open(file_path).convert('RGB')
        
        image_count += 1
        
        print("Processing image", image_count, "|| Filename: ", fname)
        
        #=================================================================
        # generate captions
        # NOTE: prompt is not needed since OPT is used
        #       if you want to use Flan-T5, follow the samples in the HuggingFace website
        inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(captions)
        
        #=================================================================
        
        # save captions
        data = {'Filename' : [fname], 'Captions' : [captions]}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(dataset_name, dataset_name + file_suffix), mode='a', index=False, header=False)
        
        #if image_count == 1: # used to test if this code is running
            #break
            
    except Exception as e:
        print("Cannot process", fname)
        continue
        
print("Total files:", image_count)

