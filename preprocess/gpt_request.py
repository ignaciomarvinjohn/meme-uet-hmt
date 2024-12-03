# gpt_request.py
# Perform requests to OpenAI through API
# Requires the meme text and caption
# NOTE: the code is quite old and may not be compatible with the current version
#       refer to the OpenAI website
#       this code is very easy to understand, so just modify the parts

import os
import sys
import pandas as pd

import openai

# get API key (refer to OpenAI website if do not have one)
openai.api_key = os.getenv("OPENAI_API_KEY")


#=================================================================
# System variables
#=================================================================

# file location of the caption
caption_file = ""

# file location of the meme text
meme_text_file = ""

# meme ID and text column name (refer to the memotion dataset)
meme_img_col = 'Id'
meme_text_col = 'ocr'

# response filename
output_file = "./dataset_name.csv"

with_logprobs = False # True if you want the logprobs and the model supports it (as of this writing, GPT-4 does not have it)


#=================================================================
# Main process
#=================================================================

# open captions dataset
df = pd.read_csv(caption_file)

print("Captions loaded.")

# open the meme text folder
meme_text_df = pd.read_csv(meme_text_file)

print("Meme texts loaded.")

# create the output file
if with_logprobs:
    out_df = pd.DataFrame(columns = ['Filename', 'Response', 'Logprobs'])
    out_df.to_csv(output_file, index=False)
else:
    out_df = pd.DataFrame(columns = ['Filename', 'Response'])
    out_df.to_csv(output_file, index=False)

#=================================================================

# for each captions
for index, sample in df.iterrows():
    print(sample['Filename'])
    
    try: 
        # get meme instance given the image
        meme_instance = meme_text_df.loc[meme_text_df[meme_img_col] == sample['Filename']]
        
        # get meme text
        meme_text = meme_instance.iloc[0][meme_text_col]
        
        # compose prompt
        prompt = 'Can you tell me the relation between the sentence "' + str(meme_text) + '" and the context "' + str(sample['Captions']) + '" in a meme image in short sentences? Also, how much do you rate your response, from 1 to 5, 1 being the lowest and 5 being the highest? Just provide a short score.'
        print(prompt)
        
        #=================================================================
        
        print("Requesting...")
        
        # make a request and get the response
        if with_logprobs:
            completion = openai.Completion.create(
                model = "davinci-002",
                prompt = prompt
            )
            
            response = completion.choices[0].text
            logprobs = completion.choices[0].logprobs
        else:
            completion = openai.ChatCompletion.create(
                model = 'gpt-4',
                messages = [
                    {'role': 'user', 'content': prompt}
                ],
                temperature = 0
            )
            response = completion['choices'][0]['message']['content']
        
        print("Done.")
        
        #=================================================================
        
        # save the response
        if with_logprobs:
            data = {'Filename' : [sample['Filename']], 'Response' : [response], 'Logprobs': [logprobs]}
            out_df = pd.DataFrame(data)
            out_df.to_csv(output_file, mode='a', index=False, header=False)
        else:
            data = {'Filename' : [sample['Filename']], 'Response' : [response]}
            out_df = pd.DataFrame(data)
            out_df.to_csv(output_file, mode='a', index=False, header=False)
            
        print("Response saved.")
        
    except Exception as e:
        print("Cannot process", sample['Filename'])
        continue
