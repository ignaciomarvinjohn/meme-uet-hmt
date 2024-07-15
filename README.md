# Meme Analysis using LLM-based Contextual Information and U-net Encapsulated Transformer
This is the main repository for the paper "Meme Analysis using LLM-based Contextual Information and U-net Encapsulated Transformer".

You can access the paper [here](https://ieeexplore.ieee.org/document/10589379).

Source code will be provided soon!

# Updates
- 2024/07/09: Paper is published in IEEE Access. Started the documentation in the GitHub.

# Setup

## Dependencies
Our work is developed in Windows 10 operating system. We used conda environment with the following dependencies:
- Python 3.7
- Cuda Toolkit 11.7
- PyTorch 1.13

## Dataset
The first stage of our work involves creating a contextual information dataset using meme images and texts. Download the Memotion [1](https://competitions.codalab.org/competitions/20629) [2](https://competitions.codalab.org/competitions/35688) [3](https://codalab.lisn.upsaclay.fr/competitions/8299) datasets and follow the instructions below on how to generate the contextual information dataset.

### Generate Meme Captions
Follow the instructions on the [BLIP-2 GitHub page](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) to set up the model for image captioning. Then, use the source code provided on the BLIP-2 website to caption the images in Memotion 1-3.

(We will provide a sample code, but keep in mind you can easily make one just by following the instructions.)

### Generate Contextual Information
Follow the instructions on the [OpenAI API website](https://platform.openai.com/docs/quickstart). You must have an account and API key (and budget) to make an API request.

Create a prompt
```
"Can you tell me the relation between the sentence "*caption*" and the context "*meme text*" in a meme image in short sentences? Also, how much do you rate your response, from 1 to 5, 1 being the lowest and 5 being the highest? Just provide a short score."
```
to generate contextual information per meme image. Note that the *meme text* in this prompt is the text in the meme image.

(We will provide a sample code, but keep in mind you can easily make one just by following the instructions.)

# Training


# Customize


# Contact
For inquiries, kindly send an email to mjci@sju.ac.kr

# Citation
```
@ARTICLE{10589379,
  author={Ignacio, Marvin John and Nguyen, Thanh Tin and Jin, Hulin and Kim, Yong-guk},
  journal={IEEE Access}, 
  title={Meme Analysis using LLM-based Contextual Information and U-net Encapsulated Transformer}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={LLM;Memes;Memotion datasets;Sentiment;Transformer;U-net},
  doi={10.1109/ACCESS.2024.3424883}}
```
