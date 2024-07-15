import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
from keybert import KeyBERT
import torch
from torch.utils.data import TensorDataset, DataLoader


class MemotionDataset:
    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        
        self.classes = None
        self.id2label = None
        self.label2id = None
        self.num_labels = None
        
    def set_info(self, classes, id2label, label2id, num_labels):
        self.classes = classes
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels


class MemotionDataloader:
    def __init__(self):
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None


# encode according to the task labels format
def label_encoding(memotion_number, memotion_dataset):
    id2label = dict()
    label2id = dict()
    classes = dict()
    num_labels = dict()
    
    if memotion_number == "memotion_1":
        # sentiment
        memotion_dataset.X_train['sentiment'].replace({'very_positive': 'positive', 'very_negative': 'negative'}, inplace=True)
        classes['sentiment'] = ['negative', 'neutral', 'positive']
        id2label['sentiment'] = {0: "negative", 1: "neutral", 2: "positive"}
        label2id['sentiment'] = {"negative": 0, "neutral": 1, "positive": 2}
        num_labels['sentiment'] = len(classes['sentiment'])
        memotion_dataset.X_train['sentiment'].replace(label2id['sentiment'], inplace=True)

        # humour
        classes['humour'] = ['not_funny', 'funny', 'very_funny', 'hilarious']
        id2label['humour'] = {0: "not_funny", 1: "funny", 2: "very_funny", 3: "hilarious"}
        label2id['humour'] = {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3}
        num_labels['humour'] = len(classes['humour'])
        memotion_dataset.X_train['humour'].replace(label2id['humour'], inplace=True)

        # sarcastic
        classes['sarcastic'] = ['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted']
        id2label['sarcastic'] = {0: "not_sarcastic", 1: "general", 2: "twisted_meaning", 3: "very_twisted"}
        label2id['sarcastic'] = {"not_sarcastic": 0, "general": 1, "twisted_meaning": 2, "very_twisted": 3}
        num_labels['sarcastic'] = len(classes['sarcastic'])
        memotion_dataset.X_train['sarcastic'].replace(label2id['sarcastic'], inplace=True)

        # offensive
        classes['offensive'] = ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive']
        id2label['offensive'] = {0: "not_offensive", 1: "slight", 2: "very_offensive", 3: "hateful_offensive"}
        label2id['offensive'] = {"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3}
        num_labels['offensive'] = len(classes['offensive'])
        memotion_dataset.X_train['offensive'].replace(label2id['offensive'], inplace=True)

        # motivational
        classes['motivational'] = ['not_motivational', 'motivational']
        id2label['motivational'] = {0: "not_motivational", 1: "motivational"}
        label2id['motivational'] = {"not_motivational": 0, "motivational": 1}
        num_labels['motivational'] = len(classes['motivational'])
        memotion_dataset.X_train['motivational'].replace(label2id['motivational'], inplace=True)

    elif memotion_number == "memotion_2":
        # sentiment
        classes['sentiment'] = ['negative', 'neutral', 'positive']
        id2label['sentiment'] = {0: "negative", 1: "neutral", 2: "positive"}
        label2id['sentiment'] = {"negative": 0, "neutral": 1, "positive": 2}
        num_labels['sentiment'] = len(classes['sentiment'])
        memotion_dataset.X_train['sentiment'].replace(label2id['sentiment'], inplace=True)
        memotion_dataset.X_val['sentiment'].replace(label2id['sentiment'], inplace=True)

        # humour
        classes['humour'] = ['not_funny', 'funny', 'very_funny', 'hilarious']
        id2label['humour'] = {0: "not_funny", 1: "funny", 2: "very_funny", 3: "hilarious"}
        label2id['humour'] = {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3}
        num_labels['humour'] = len(classes['humour'])
        memotion_dataset.X_train['humour'].replace(label2id['humour'], inplace=True)
        memotion_dataset.X_val['humour'].replace(label2id['humour'], inplace=True)

        # sarcastic
        classes['sarcastic'] = ['not_sarcastic', 'little_sarcastic', 'very_sarcastic', 'extremely_sarcastic']
        id2label['sarcastic'] = {0: "not_sarcastic", 1: "little_sarcastic", 2: "very_sarcastic", 3: "extremely_sarcastic"}
        label2id['sarcastic'] = {"not_sarcastic": 0, "little_sarcastic": 1, "very_sarcastic": 2, "extremely_sarcastic": 3}
        num_labels['sarcastic'] = len(classes['sarcastic'])
        memotion_dataset.X_train['sarcastic'].replace(label2id['sarcastic'], inplace=True)
        memotion_dataset.X_val['sarcastic'].replace(label2id['sarcastic'], inplace=True)

        # offensive
        classes['offensive'] = ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive']
        id2label['offensive'] = {0: "not_offensive", 1: "slight", 2: "very_offensive", 3: "hateful_offensive"}
        label2id['offensive'] = {"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3}
        num_labels['offensive'] = len(classes['offensive'])
        memotion_dataset.X_train['offensive'].replace(label2id['offensive'], inplace=True)
        memotion_dataset.X_val['offensive'].replace(label2id['offensive'], inplace=True)

        # motivational
        classes['motivational'] = ['not_motivational', 'motivational']
        id2label['motivational'] = {0: "not_motivational", 1: "motivational"}
        label2id['motivational'] = {"not_motivational": 0, "motivational": 1}
        num_labels['motivational'] = len(classes['motivational'])
        memotion_dataset.X_train['motivational'].replace(label2id['motivational'], inplace=True)
        memotion_dataset.X_val['motivational'].replace(label2id['motivational'], inplace=True)

    elif memotion_number == "memotion_3":
        # sentiment
        memotion_dataset.X_train['sentiment'].replace({'very_positive': 'positive', 'very_negative': 'negative'}, inplace=True)
        memotion_dataset.X_val['sentiment'].replace({'very_positive': 'positive', 'very_negative': 'negative'}, inplace=True)
        classes['sentiment'] = ['negative', 'neutral', 'positive']
        id2label['sentiment'] = {0: "negative", 1: "neutral", 2: "positive"}
        label2id['sentiment'] = {"negative": 0, "neutral": 1, "positive": 2}
        num_labels['sentiment'] = len(classes['sentiment'])
        memotion_dataset.X_train['sentiment'].replace(label2id['sentiment'], inplace=True)
        memotion_dataset.X_val['sentiment'].replace(label2id['sentiment'], inplace=True)

        # humour
        classes['humour'] = ['not_funny', 'funny', 'very_funny', 'hilarious']
        id2label['humour'] = {0: "not_funny", 1: "funny", 2: "very_funny", 3: "hilarious"}
        label2id['humour'] = {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3}
        num_labels['humour'] = len(classes['humour'])
        memotion_dataset.X_train['humour'].replace(label2id['humour'], inplace=True)
        memotion_dataset.X_val['humour'].replace(label2id['humour'], inplace=True)

        # sarcastic
        classes['sarcastic'] = ['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted']
        id2label['sarcastic'] = {0: "not_sarcastic", 1: "general", 2: "twisted_meaning", 3: "very_twisted"}
        label2id['sarcastic'] = {"not_sarcastic": 0, "general": 1, "twisted_meaning": 2, "very_twisted": 3}
        num_labels['sarcastic'] = len(classes['sarcastic'])
        memotion_dataset.X_train['sarcastic'].replace(label2id['sarcastic'], inplace=True)
        memotion_dataset.X_val['sarcastic'].replace(label2id['sarcastic'], inplace=True)

        # offensive
        classes['offensive'] = ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive']
        id2label['offensive'] = {0: "not_offensive", 1: "slight", 2: "very_offensive", 3: "hateful_offensive"}
        label2id['offensive'] = {"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3}
        num_labels['offensive'] = len(classes['offensive'])
        memotion_dataset.X_train['offensive'].replace(label2id['offensive'], inplace=True)
        memotion_dataset.X_val['offensive'].replace(label2id['offensive'], inplace=True)

        # motivational
        classes['motivational'] = ['not_motivational', 'motivational']
        id2label['motivational'] = {0: "not_motivational", 1: "motivational"}
        label2id['motivational'] = {"not_motivational": 0, "motivational": 1}
        num_labels['motivational'] = len(classes['motivational'])
        memotion_dataset.X_train['motivational'].replace(label2id['motivational'], inplace=True)
        memotion_dataset.X_val['motivational'].replace(label2id['motivational'], inplace=True)

    else:
        print("Encoding error. Program exiting...")
        quit()

    memotion_dataset.set_info(classes, id2label, label2id, num_labels)

    # return all required variables
    return memotion_dataset


# get the document and keyword embeddings
# since the embedding size is the same per type, we add the document embedding in the last entry
def get_doc_key_embeddings(X, n_grams, num_keywords):
    # initialize model
    kw_model = KeyBERT(model="all-mpnet-base-v2")
    
    print("Extracting keywords...")
    
    # extract keywords
    keywords_all = kw_model.extract_keywords(X.values, keyphrase_ngram_range=n_grams, top_n=num_keywords)
    
    print("Done. Getting the embeddings...")
    
    all_embeddings = []
    doc_embeddings = None
    
    progress_bar = tqdm(range(len(X)))
    
    # for each sample
    for i in range(len(X)):
        temp = []
        
        # get the strings
        for keyword in keywords_all[i]:
            temp.append(keyword[0])
            
        doc_embeddings, word_embeddings = kw_model.extract_embeddings(X.values[i], temp)
        
        temp_size = word_embeddings.shape[1]
        
        # perform padding if the keywords are less than the number of keywords needed
        if len(word_embeddings) < num_keywords:
            
            num_need = num_keywords - len(word_embeddings)
            
            for i in range(num_need):
                pad_emb = np.zeros((1, temp_size))
                word_embeddings = np.concatenate((word_embeddings, pad_emb), axis=0)
                
        word_embeddings = np.concatenate((word_embeddings, doc_embeddings), axis=0)
        all_embeddings.append(word_embeddings)
        
        progress_bar.update(1)
        
    progress_bar.clear()
    progress_bar.close()
    
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    
    print("Done.")
    
    return all_embeddings


# get the document embeddings
def get_doc_embeddings(X, n_grams, num_keywords):
    # initialize model
    kw_model = KeyBERT(model="all-mpnet-base-v2")
    
    print("Extracting keywords...")
    
    # extract keywords
    keywords_all = kw_model.extract_keywords(X.values, keyphrase_ngram_range=n_grams, top_n=num_keywords)
    
    print("Done. Getting the embeddings...")
    
    all_embeddings = []
    doc_embeddings = None
    
    progress_bar = tqdm(range(len(X)))
    
    # for each sample
    for i in range(len(X)):
        temp = []
        
        # get the strings
        for keyword in keywords_all[i]:
            temp.append(keyword[0])
            
        doc_embeddings, word_embeddings = kw_model.extract_embeddings(X.values[i], temp)
        
        # save document embeddings
        all_embeddings.append(doc_embeddings)
        
        progress_bar.update(1)
        
    progress_bar.clear()
    progress_bar.close()
    
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    
    print("Done.")
    
    return all_embeddings


# function to increase the sample of the class group that is less than 1
# in effect, stratified split will have 1 sample in each partition
# used only in Memotion 1 since it doesn't have a validation set
def populate_train_val(df, tasks_list):
    
    df_groups = df.groupby(tasks_list)
    
    ids_list = []
    
    for name, group in df_groups:
        if group['Id'].count() == 1:
            ids_list.append(group['Id'].values[0])
            
    print("There are " + str(len(ids_list)) + " groups with single entries. Duplicating...")
    
    duplicates = df[df['Id'].isin(ids_list)]
    df = df.append(duplicates, ignore_index=True)
    
    print("Done.")
    
    return df


# converts the dataset into dataloader
def to_dataloader(memotion_dataset, batch_size):
    # create a memotion dataloader object
    memotion_dataloader = MemotionDataloader()
    
    # dataloader for training set
    X_train_tensor = torch.Tensor(memotion_dataset.X_train)
    y_train_tensor = torch.Tensor(memotion_dataset.y_train.values)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    memotion_dataloader.train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    
    # dataloader for validation set
    X_val_tensor = torch.Tensor(memotion_dataset.X_val)
    y_val_tensor = torch.Tensor(memotion_dataset.y_val.values)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    memotion_dataloader.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # dataloader for test set
    X_test_tensor = torch.Tensor(memotion_dataset.X_test)
    
    test_dataset = TensorDataset(X_test_tensor)
    memotion_dataloader.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("------------------------------")
    print("X_train tensor type", type(X_train_tensor))
    print("y_train tensor type", type(y_train_tensor))
    print("X_val tensor type", type(X_val_tensor))
    print("y_val tensor type", type(y_val_tensor))
    print("X_test tensor type", type(X_test_tensor))
    print("------------------------------")
    print("X_train tensor shape", X_train_tensor.shape)
    print("y_train tensor shape", y_train_tensor.shape)
    print("X_val tensor shape", X_val_tensor.shape)
    print("y_val tensor shape", y_val_tensor.shape)
    print("X_test tensor shape", X_test_tensor.shape)
    print("------------------------------")
    
    return memotion_dataloader


def make_partition_numbers(num_models):
    set_numbers_list = []
    random.seed(42)
    
    # make a random sampling of partition numbers (w/repeat), but make the whole ensemble contain all partitions
    while True:

        unique_nums = []
        
        # for each model, make a random sampling of partition numbers
        for num in range(num_models):
            
            # randomize partition and get unique partition number
            temp_list = [random.randint(1,10) for i in range(10)]
            set_numbers_list.append(temp_list)
            unique_nums.extend(list(set(temp_list)))
            unique_nums = list(set(unique_nums))
            
            
        # if all partition numbers are used, exit loop
        if unique_nums == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print("Partitions list:")
            print(set_numbers_list)
            # print(unique_nums)
            
            break
            
    return set_numbers_list


# create a bootstrap dataset (for bagging algorithm)
def make_bootstrap_datasets(memotion_dataset, sampling_list):
    
    # calculate how many samples per set
    n_samples_per_set = int(len(memotion_dataset.X_train)/10)
    
    # create a temporary dataset
    temp_ds = MemotionDataset()
    temp_ds.X_train = memotion_dataset.X_train
    temp_ds.y_train = memotion_dataset.y_train
    
    # get the partitions according to partition number
    for i in range(10):
        start_sample = n_samples_per_set * i
        end_sample = n_samples_per_set * (i+1)
        
        print("Start, end:", start_sample, end_sample)
        
        sampling_index = sampling_list[i]
        start_orig = n_samples_per_set * (sampling_index-1)
        end_orig = n_samples_per_set * sampling_index
        
        print("Start, end:", start_orig, end_orig)
        
        temp_ds.X_train[ start_sample:end_sample , : ] = memotion_dataset.X_train[ start_orig:end_orig , : ]
        temp_ds.y_train.iloc[start_sample:end_sample] = memotion_dataset.y_train.iloc[start_orig:end_orig]
        
    temp_ds.X_val = memotion_dataset.X_val
    temp_ds.y_val = memotion_dataset.y_val
    temp_ds.X_test = memotion_dataset.X_test
    temp_ds.set_info(memotion_dataset.classes, memotion_dataset.id2label, memotion_dataset.label2id, memotion_dataset.num_labels)
    
    
    return temp_ds


def get_test_labels(memotion_number):
    
    all_labels = {'ta_sen':[], 'tb_hum':[], 'tb_sar':[], 'tb_off':[], 'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
    
    # get labels depending on the memotion number
    if memotion_number == "memotion_1":
        all_labels = get_m1_test_labels(all_labels)
    elif memotion_number == "memotion_2":
        all_labels = get_m2_test_labels(all_labels)
    elif memotion_number == "memotion_3":
        all_labels = get_m3_test_labels(all_labels)
    else:
        print("Encoding error. Program exiting...")
        quit()
        
    return all_labels


def get_m1_test_labels(all_labels):
    # open file
    gt_df = pd.read_csv("./input/labels/memotion_1_test_gt.csv")
    
    # for each line, parse labels and store
    for index, gt_sample in gt_df.iterrows():
        gt_labels = gt_sample['Labels']
        gt_tasks_labels = gt_labels.split("_")
        
        # task A
        all_labels['ta_sen'].append(int(gt_tasks_labels[0]))
        
        # task B
        gt_tB_label = [*gt_tasks_labels[1]]
        all_labels['tb_hum'].append(int(gt_tB_label[0]))
        all_labels['tb_sar'].append(int(gt_tB_label[1]))
        all_labels['tb_off'].append(int(gt_tB_label[2]))
        all_labels['tb_mot'].append(int(gt_tB_label[3]))
        
        # task C
        gt_tC_label = [*gt_tasks_labels[2]]
        all_labels['tc_hum'].append(int(gt_tC_label[0]))
        all_labels['tc_sar'].append(int(gt_tC_label[1]))
        all_labels['tc_off'].append(int(gt_tC_label[2]))
        
    return all_labels


def get_m2_test_labels(all_labels):
    # open file
    gt_df = pd.read_csv("./input/labels/memotion_2_test_gt.csv")
    
    # replace values in the ground truth file
    gt_df['overall_sentiment'].replace({"negative": 0, "neutral": 1, "positive": 2}, inplace=True)
    gt_df['humour'].replace({"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3}, inplace=True)
    gt_df['sarcastic'].replace({"not_sarcastic": 0, "little_sarcastic": 1, "very_sarcastic": 2, "extremely_sarcastic": 3}, inplace=True)
    gt_df['offensive'].replace({"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3}, inplace=True)
    gt_df['motivational'].replace({"not_motivational": 0, "motivational": 1}, inplace=True)
    
    # for each line, parse labels and store
    for index, gt_sample in gt_df.iterrows():
    
        # task A
        all_labels['ta_sen'].append(int(gt_sample['overall_sentiment']))
        
        # task B
        all_labels['tb_hum'].append(1 if gt_sample['humour'] > 0 else 0)
        all_labels['tb_sar'].append(1 if gt_sample['sarcastic'] > 0 else 0)
        all_labels['tb_off'].append(1 if gt_sample['offensive'] > 0 else 0)
        all_labels['tb_mot'].append(1 if gt_sample['motivational'] > 0 else 0)
        
        # task C
        all_labels['tc_hum'].append(int(gt_sample['humour']))
        all_labels['tc_sar'].append(int(gt_sample['sarcastic']))
        all_labels['tc_off'].append(int(gt_sample['offensive']))
        
    return all_labels


def get_m3_test_labels(all_labels):
    # open file
    gt_df = pd.read_csv("./input/labels/memotion_3_test_gt.txt", header=None)
    
    # for each line, parse labels and store
    for index, gt_labels in gt_df[0].items():
        gt_tasks_labels = gt_labels.split("_")
        
        # task A
        all_labels['ta_sen'].append(int(gt_tasks_labels[0]))
        
        # task B
        gt_tB_label = [*gt_tasks_labels[1]]
        all_labels['tb_hum'].append(int(gt_tB_label[0]))
        all_labels['tb_sar'].append(int(gt_tB_label[1]))
        all_labels['tb_off'].append(int(gt_tB_label[2]))
        all_labels['tb_mot'].append(int(gt_tB_label[3]))
        
        # task C
        gt_tC_label = [*gt_tasks_labels[2]]
        all_labels['tc_hum'].append(int(gt_tC_label[0]))
        all_labels['tc_sar'].append(int(gt_tC_label[1]))
        all_labels['tc_off'].append(int(gt_tC_label[2]))
        
    return all_labels

