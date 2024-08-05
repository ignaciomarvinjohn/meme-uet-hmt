# Meme Analysis using LLM-based Contextual Information and U-net Encapsulated Transformer
This is the main repository for the paper "Meme Analysis using LLM-based Contextual Information and U-net Encapsulated Transformer".

You can access the paper [here](https://ieeexplore.ieee.org/document/10589379).

# Updates
- 2024/07/09: Paper is published in IEEE Access.
- 2024/07/15: Base code is uploaded to GitHub.

# Setup

## Dependencies
Our work is developed using the Windows 10 operating system. We used a conda environment with the following dependencies:
- Python 3.7
- Cuda Toolkit 11.7
- PyTorch 1.13

## Dataset
The first stage of our work involves creating a contextual information dataset using meme images and texts. Download the Memotion [1](https://competitions.codalab.org/competitions/20629) [2](https://competitions.codalab.org/competitions/35688) [3](https://codalab.lisn.upsaclay.fr/competitions/8299) datasets and follow the instructions below on how to generate the contextual information dataset.

### Generate Meme Captions
Follow the instructions on the [BLIP-2 GitHub page](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) to set up the model for image captioning. Then, use the source code provided on the BLIP-2 website to caption the images in Memotion 1-3.

(We will provide a sample code, but you can easily make one by following the instructions.)

### Generate Contextual Information
Follow the instructions on the [OpenAI API website](https://platform.openai.com/docs/quickstart). You must have an account and API key (and budget) to make an API request. In your code, use GPT-4 as the model.

Create a prompt
```
"Can you tell me the relation between the sentence "meme_caption" and the context "meme_text" in a meme image in short sentences? Also, how much do you rate your response, from 1 to 5, 1 being the lowest and 5 being the highest? Just provide a short score."
```
to generate contextual information per meme image. Note that the *meme_text* in this prompt is the text in the meme image, and *meme_caption* is the generated caption using the BLIP-2 model.

(We will provide a sample code, but you can easily make one by following the instructions.)

# Training
We have provided our own contextual information dataset, which is located in the *input* folder. ATTENTION: This dataset was built at our own expense. We originally intended to make it private, but we decided to make it public for the sake of research. Take note that this dataset is under the [**CC BY-NC-ND 4.0**](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

You can run the following code to perform classification using Document HMT, as described in our paper.
```
python train_document.py
```
This performs classification without the UET. It utilizes the document embeddings extracted using KeyBERT.

To classify the contextual information using Keywords HMT, run the following code.
```
python train_keywords.py
```
This is the version that uses UET, which utilizes the keywords/keyphrases.

Our code does not use the command line. Instead, the experiment can be changed using control parameters and hyperparameters.

Control Parameters:
- is_debug = Use this for debugging (checking the dimensions at a certain point).
- is_train = True if you want to train and test an experiment; False if you want to test an experiment (you need to have a model first).
- memotion_num_list = Tells which Memotion dataset you want to run. You can simply add/remove the dataset(s) here.
- tasks_list = The Memotion tasks. Our prior code (single task) utilizes this to select a specific task for a particular experiment.
- version = Also called "experiment number". Change this every time you make an experiment.

Refer to the paper for the hyperparameters.

# Customization
Our research is composed of different modules that other algorithms can easily substitute. We encourage you to try out other models and see their performance.

You can:
1. Change BLIP-2 using a different image captioning model.
2. Use a different LLM other than GPT-4. (I suggest using an open-source LLM, which does not require financial cost.)
3. Craft your own prompt.
4. Create your own dataset using another method.

Our code is also written in a comprehensive way so that you can edit them yourself.

# Notes
If you have concerns or suggestions regarding our GitHub, don't hesitate to message us. We want to improve this as much as possible, so your comments are welcome!

For inquiries, kindly send an email to mjci@sju.ac.kr.

# Other Links
- **UET4Rec**: U-net encapsulated transformer for sequential recommender | [paper](https://www.sciencedirect.com/science/article/pii/S0957417424016488)
- **VisUET** | [Github](https://github.com/ignaciomarvinjohn/visuet)

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
