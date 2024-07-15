#=========================================================================
# Keywords Embedding HMT Ensemble
#=========================================================================
import os
import shutil
import pandas as pd
import numpy as np
import time
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from scripts.libs import dataset
from scripts.libs.model import UET, TaskAPredictLayerH, TaskBPredictLayerH, TaskCPredictLayerH
from scripts.utils import display


# control parameters
is_debug = False
is_train = True
memotion_num_list = [1,2,3]
tasks_list = ['sentiment','humour','sarcastic','offensive','motivational']
version = "key_1"

# hyperparameters
n_grams = (1,6)
num_keywords = 10
batch_size = 2048
num_epochs = 200
learning_rate = 0.05
n_ensemble = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# get positive class weight (binary class)
def get_positive_weight(memotion_dataset, task):
    class_counts_dict = memotion_dataset.y_train[task].value_counts().to_dict()
    
    # get value counts for the negative class
    neg_weight = class_counts_dict[0]
    
    # get value counts for the rest (positive class)
    pos_weight = 0
    for i in range(1, len(class_counts_dict)):
        pos_weight += class_counts_dict[i]
        
    return neg_weight / pos_weight


# get weights for the loss (multi-class)
def get_class_weights(memotion_dataset, task):
    class_weights = []
    class_counts_dict = memotion_dataset.y_train[task].value_counts().to_dict()
    for i in range(len(class_counts_dict)):
        class_weights.append(memotion_dataset.y_train[task].shape[0] / class_counts_dict[i])
        
    return class_weights


# create model
# since we separate the modules, we make a dictionary
def create_model(memotion_dataset):
    # get embedding size
    emb_size = memotion_dataset.X_train.shape[2]
    print("Embedding size:", emb_size)
    
    emb_sizes = []
    emb_sizes.append(int(emb_size))
    emb_sizes.append(int(emb_size/2))
    emb_sizes.append(int(emb_size/4))
    emb_sizes.append(int(emb_size/8))
    
    in_size = []
    in_size.append(int(num_keywords * emb_size))
    in_size.append(1024)
    in_size.append(512)
    # in_size.append(256)
    
    # create a model dictionary
    model = dict()
    
    # add feature extraction module
    model['base'] = UET(emb_sizes)
    # model['base'] = PassLayer()
    model['base'].to(device)
    
    # add sentiment predictor
    model['ta_sen'] = TaskAPredictLayerH(in_size[0], in_size[1:], 4, memotion_dataset.num_labels['sentiment'])
    model['ta_sen'].to(device)
    
    # add emotion predictor
    for part in ['tb_hum', 'tb_sar', 'tb_off', 'tb_mot']:
        model[part] = TaskBPredictLayerH(in_size[0], in_size[1:], 1)
        model[part].to(device)
        
    # add intensity predictor
    for part, task in zip(['tc_hum', 'tc_sar', 'tc_off'], ['humour', 'sarcastic', 'offensive']):
        model[part] = TaskCPredictLayerH(in_size[0], in_size[1:], memotion_dataset.num_labels[task])
        model[part].to(device)
        
    # add binary cross-entropy loss functions
    for key, task in zip(['hum_bin_loss', 'sar_bin_loss', 'off_bin_loss', 'mot_bin_loss'], ['humour', 'sarcastic', 'offensive', 'motivational']):
        model[key] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(get_positive_weight(memotion_dataset, task))).to(device)
        
    # add cross-entropy loss functions
    for key, task in zip(['sen_loss', 'hum_loss', 'sar_loss', 'off_loss'], ['sentiment', 'humour', 'sarcastic', 'offensive']):
        model[key] = nn.CrossEntropyLoss(weight=torch.tensor(get_class_weights(memotion_dataset, task))).to(device)
        
        
    return model


# get model parameters for optimizer
def get_model_params(model, model_parts):
    params = list()
    
    for part in model_parts:
        params += list(model[part].parameters())
    
    return params


# change the mode of multiple model parts
def mode_change(model, model_parts, mode):
    
    for part in model_parts:
        if mode == "train":
            model[part] = model[part].train()
        else:
            model[part] = model[part].eval()
            
    return model


# get predictions and formatted labels
def get_predictions_and_labels(model, model_part, logits, labels, is_binary=False):
    # detach gradient
    with torch.no_grad():
        labels = labels.numpy().astype(int)
        
        # predict and compute for the loss
        preds = model[model_part](logits)
        
        if is_binary:
            preds = torch.sigmoid(preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            preds = torch.squeeze(preds)
            preds = preds.cpu().numpy().astype(int)
        else:
            preds = torch.argmax(preds, dim=-1)
            preds = preds.cpu().numpy()
    
    return preds, labels


# get predictions and formatted labels (hierarchical)
def get_predictions_and_labels_c(model, model_part, logits, labels, bin_preds):
    # detach gradient
    with torch.no_grad():
        labels = labels.numpy().astype(int)
        
        # predict and compute for the loss
        preds = model[model_part](logits, bin_preds)
        
        preds = torch.argmax(preds, dim=-1)
        preds = preds.cpu().numpy()
        
    return preds, labels


# get predictions
def get_predictions(model, model_part, logits, is_binary=False):
    # detach gradient
    with torch.no_grad():
        
        # predict and compute for the loss
        preds = model[model_part](logits)
        
        if is_binary:
            preds = torch.sigmoid(preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            preds = torch.squeeze(preds)
            preds = preds.cpu().numpy().astype(int)
        else:
            preds = torch.argmax(preds, dim=-1)
            preds = preds.cpu().numpy()
            
    return preds


# get predictions for task C (hierarchical)
def get_predictions_c(model, model_part, logits, bin_preds):
    # detach gradient
    with torch.no_grad():
        
        # predict and compute for the loss
        preds = model[model_part](logits, bin_preds)
        
        preds = torch.argmax(preds, dim=-1)
        preds = preds.cpu().numpy()
            
    return preds


# make predictions and return the loss function
def make_predictions(model, model_part, loss_fn, logits, labels, is_binary=False):
    # prepare labels
    if is_binary:
        labels = labels.unsqueeze(1)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)
    else:
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        
    # predict and compute for the loss
    preds = model[model_part](logits)
    loss = loss_fn(preds, labels)
    
    return loss


# make predictions for task C (hierarchical)
def make_predictions_c(model, model_part, loss_fn, logits, labels, bin_preds):
    # prepare labels
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)
        
    # predict and compute for the loss
    preds = model[model_part](logits, bin_preds)
    loss = loss_fn(preds, labels)
    
    return loss

def train_emotion(memotion_number, model, memotion_dataloader):
    params = get_model_params(model, ['base', 'tb_hum', 'tb_sar', 'tb_off', 'tb_mot'])
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    train_loss_history = {'humour': [], 'sarcastic': [], 'offensive': [], 'motivational': []}
    valid_loss_history = {'humour': [], 'sarcastic': [], 'offensive': [], 'motivational': []}
    data_size = dict()
    data_size['train'] = len(memotion_dataloader.train_dataloader)
    data_size['val'] = len(memotion_dataloader.val_dataloader)
    current_tasks = ['humour','sarcastic','offensive','motivational']
    
    # for each epoch
    for e in range(num_epochs):
        
        # train the model
        model = mode_change(model, ['base', 'tb_hum', 'tb_sar', 'tb_off', 'tb_mot'], "train")
        train_loss = {'humour': 0.0, 'sarcastic': 0.0, 'offensive': 0.0, 'motivational': 0.0}
        for data, labels in memotion_dataloader.train_dataloader:
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            # convert labels to binary
            labels[labels > 0] = 1
            
            hum_loss = make_predictions(model, 'tb_hum', model['hum_bin_loss'], logits, labels[ : , 1], True)
            sar_loss = make_predictions(model, 'tb_sar', model['sar_bin_loss'], logits, labels[ : , 2], True)
            off_loss = make_predictions(model, 'tb_off', model['off_bin_loss'], logits, labels[ : , 3], True)
            mot_loss = make_predictions(model, 'tb_mot', model['mot_bin_loss'], logits, labels[ : , 4], True)
            
            train_loss['humour'] += hum_loss.item()
            train_loss['sarcastic'] += sar_loss.item()
            train_loss['offensive'] += off_loss.item()
            train_loss['motivational'] += mot_loss.item()
            
            loss = hum_loss + sar_loss + off_loss + mot_loss
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # store all training loss per task
        for task in current_tasks:
            train_loss_history[task].append(train_loss[task])
            
            
        # perform validation
        model = mode_change(model, ['base', 'tb_hum', 'tb_sar', 'tb_off', 'tb_mot'], "eval")
        valid_loss = {'humour': 0.0, 'sarcastic': 0.0, 'offensive': 0.0, 'motivational': 0.0}
        for data, labels in memotion_dataloader.val_dataloader:
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            # convert labels to binary
            labels[labels > 0] = 1
            
            hum_loss = make_predictions(model, 'tb_hum', model['hum_bin_loss'], logits, labels[ : , 1], True)
            sar_loss = make_predictions(model, 'tb_sar', model['sar_bin_loss'], logits, labels[ : , 2], True)
            off_loss = make_predictions(model, 'tb_off', model['off_bin_loss'], logits, labels[ : , 3], True)
            mot_loss = make_predictions(model, 'tb_mot', model['mot_bin_loss'], logits, labels[ : , 4], True)
            
            valid_loss['humour'] += hum_loss.item()
            valid_loss['sarcastic'] += sar_loss.item()
            valid_loss['offensive'] += off_loss.item()
            valid_loss['motivational'] += mot_loss.item()
            
        # store all validation loss per task
        for task in current_tasks:
            valid_loss_history[task].append(valid_loss[task])
            
            
        # display loss values
        display.loss_per_epoch(e, data_size, train_loss_history, valid_loss_history, current_tasks)
        
    # plot history and save
    display.make_loss_plot(memotion_number + "_v" + version, "task_b", train_loss_history, valid_loss_history, current_tasks)
    
    return model


def train_intensity(memotion_number, model, memotion_dataloader):
    params = get_model_params(model, ['base', 'tc_hum', 'tc_sar', 'tc_off'])
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    train_loss_history = {'humour': [], 'sarcastic': [], 'offensive': []}
    valid_loss_history = {'humour': [], 'sarcastic': [], 'offensive': []}
    data_size = dict()
    data_size['train'] = len(memotion_dataloader.train_dataloader)
    data_size['val'] = len(memotion_dataloader.val_dataloader)
    current_tasks = ['humour','sarcastic','offensive']
    
    # for each epoch
    for e in range(num_epochs):
        
        # train the model
        model = mode_change(model, ['base', 'tc_hum', 'tc_sar', 'tc_off'], "train")
        model = mode_change(model, ['tb_hum', 'tb_sar', 'tb_off'], "eval")
        train_loss = {'humour': 0.0, 'sarcastic': 0.0, 'offensive': 0.0}
        for data, labels in memotion_dataloader.train_dataloader:
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            tb_hum_preds = get_predictions(model, 'tb_hum', logits, is_binary=True)
            tb_sar_preds = get_predictions(model, 'tb_sar', logits, is_binary=True)
            tb_off_preds = get_predictions(model, 'tb_off', logits, is_binary=True)
            
            hum_loss = make_predictions_c(model, 'tc_hum', model['hum_loss'], logits, labels[ : , 1], tb_hum_preds)
            sar_loss = make_predictions_c(model, 'tc_sar', model['sar_loss'], logits, labels[ : , 2], tb_sar_preds)
            off_loss = make_predictions_c(model, 'tc_off', model['off_loss'], logits, labels[ : , 3], tb_off_preds)
            
            train_loss['humour'] += hum_loss.item()
            train_loss['sarcastic'] += sar_loss.item()
            train_loss['offensive'] += off_loss.item()
            
            loss = hum_loss + sar_loss + off_loss
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # store all training loss per task
        for task in current_tasks:
            train_loss_history[task].append(train_loss[task])
            
            
        # perform validation
        model = mode_change(model, ['base', 'tc_hum', 'tc_sar', 'tc_off'], "eval")
        valid_loss = {'humour': 0.0, 'sarcastic': 0.0, 'offensive': 0.0}
        for data, labels in memotion_dataloader.val_dataloader:
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            tb_hum_preds = get_predictions(model, 'tb_hum', logits, is_binary=True)
            tb_sar_preds = get_predictions(model, 'tb_sar', logits, is_binary=True)
            tb_off_preds = get_predictions(model, 'tb_off', logits, is_binary=True)
            
            hum_loss = make_predictions_c(model, 'tc_hum', model['hum_loss'], logits, labels[ : , 1], tb_hum_preds)
            sar_loss = make_predictions_c(model, 'tc_sar', model['sar_loss'], logits, labels[ : , 2], tb_sar_preds)
            off_loss = make_predictions_c(model, 'tc_off', model['off_loss'], logits, labels[ : , 3], tb_off_preds)
            
            valid_loss['humour'] += hum_loss.item()
            valid_loss['sarcastic'] += sar_loss.item()
            valid_loss['offensive'] += off_loss.item()
            
        # store all validation loss per task
        for task in current_tasks:
            valid_loss_history[task].append(valid_loss[task])
            
            
        # display loss values
        display.loss_per_epoch(e, data_size, train_loss_history, valid_loss_history, current_tasks)
        
    # plot history and save
    display.make_loss_plot(memotion_number + "_v" + version, "task_c", train_loss_history, valid_loss_history, current_tasks)
    
    return model


def train_sentiment(memotion_number, model, memotion_dataloader):
    params = get_model_params(model, ['base', 'ta_sen'])
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    train_loss_history = {'sentiment': []}
    valid_loss_history = {'sentiment': []}
    data_size = dict()
    data_size['train'] = len(memotion_dataloader.train_dataloader)
    data_size['val'] = len(memotion_dataloader.val_dataloader)
    current_tasks = ['sentiment']
    
    # for each epoch
    for e in range(num_epochs):
        
        # train the model
        model = mode_change(model, ['base', 'ta_sen'], "train")
        train_loss = {'sentiment': 0.0}
        for data, labels in memotion_dataloader.train_dataloader:
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            tb_hum_preds = get_predictions(model, 'tb_hum', logits, is_binary=True)
            tb_sar_preds = get_predictions(model, 'tb_sar', logits, is_binary=True)
            tb_off_preds = get_predictions(model, 'tb_off', logits, is_binary=True)
            tb_mot_preds = get_predictions(model, 'tb_mot', logits, is_binary=True)
            
            tc_hum_preds = get_predictions_c(model, 'tc_hum', logits, tb_hum_preds)
            tc_sar_preds = get_predictions_c(model, 'tc_sar', logits, tb_sar_preds)
            tc_off_preds = get_predictions_c(model, 'tc_off', logits, tb_off_preds)
            
            tc_hum_preds = torch.Tensor(tc_hum_preds).to(device)
            tc_sar_preds = torch.Tensor(tc_sar_preds).to(device)
            tc_off_preds = torch.Tensor(tc_off_preds).to(device)
            tb_mot_preds = torch.Tensor(tb_mot_preds).to(device)
            
            # flatten logits
            # B x N x E -> B x N*E
            logits = torch.flatten(logits, 1)
            
            logits = torch.cat((logits, torch.unsqueeze(tc_hum_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_sar_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_off_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tb_mot_preds, 1)), dim=1)
            
            # B x (N*E) + 4
            sen_loss = make_predictions(model, 'ta_sen', model['sen_loss'], logits, labels[ : , 0])
            
            train_loss['sentiment'] += sen_loss.item()
            
            loss = sen_loss
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # store all training loss per task
        for task in current_tasks:
            train_loss_history[task].append(train_loss[task])
            
            
        # perform validation
        model = mode_change(model, ['base', 'ta_sen'], "eval")
        valid_loss = {'sentiment': 0.0}
        for data, labels in memotion_dataloader.val_dataloader:
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            tb_hum_preds = get_predictions(model, 'tb_hum', logits, is_binary=True)
            tb_sar_preds = get_predictions(model, 'tb_sar', logits, is_binary=True)
            tb_off_preds = get_predictions(model, 'tb_off', logits, is_binary=True)
            tb_mot_preds = get_predictions(model, 'tb_mot', logits, is_binary=True)
            
            tc_hum_preds = get_predictions_c(model, 'tc_hum', logits, tb_hum_preds)
            tc_sar_preds = get_predictions_c(model, 'tc_sar', logits, tb_sar_preds)
            tc_off_preds = get_predictions_c(model, 'tc_off', logits, tb_off_preds)
            
            tc_hum_preds = torch.Tensor(tc_hum_preds).to(device)
            tc_sar_preds = torch.Tensor(tc_sar_preds).to(device)
            tc_off_preds = torch.Tensor(tc_off_preds).to(device)
            tb_mot_preds = torch.Tensor(tb_mot_preds).to(device)
            
            # flatten logits
            # B x N x E -> B x N*E
            logits = torch.flatten(logits, 1)
            
            logits = torch.cat((logits, torch.unsqueeze(tc_hum_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_sar_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_off_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tb_mot_preds, 1)), dim=1)
            
            # B x (N*E) + 4
            sen_loss = make_predictions(model, 'ta_sen', model['sen_loss'], logits, labels[ : , 0])
            
            valid_loss['sentiment'] += sen_loss.item()
            
        # store all validation loss per task
        for task in current_tasks:
            valid_loss_history[task].append(valid_loss[task])
            
            
        # display loss values
        display.loss_per_epoch(e, data_size, train_loss_history, valid_loss_history, current_tasks)
        
    # plot history and save
    display.make_loss_plot(memotion_number + "_v" + version, "task_a", train_loss_history, valid_loss_history, current_tasks)
    
    return model


def validate_model(memotion_number, model, memotion_dataloader):
    
    all_preds = {'ta_sen':[], 'tb_hum':[], 'tb_sar':[], 'tb_off':[], 'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
    all_labels = {'ta_sen':[], 'tb_hum':[], 'tb_sar':[], 'tb_off':[], 'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
    
    # eval mode
    model_parts = ['base', 'ta_sen', 'tb_hum', 'tb_sar', 'tb_off', 'tb_mot', 'tc_hum', 'tc_sar', 'tc_off']
    model = mode_change(model, model_parts, "eval")
    
    with torch.no_grad():
        # for each batch in validation set
        for data, labels in memotion_dataloader.val_dataloader:
            
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            # convert labels to binary
            bin_labels = labels
            bin_labels[bin_labels > 0] = 1
            
            temp = {'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
            
            # predict for task B and C
            for taskb, taskc, num in zip(['tb_hum','tb_sar','tb_off','tb_mot'], ['tc_hum','tc_sar','tc_off', 'blank'], [1,2,3,4]):
                b_preds, b_labels = get_predictions_and_labels(model, taskb, logits, bin_labels[ : , num], is_binary=True)
                all_preds[taskb].extend(b_preds)
                all_labels[taskb].extend(b_labels)
                
                if num < 4:
                    c_preds, c_labels = get_predictions_and_labels_c(model, taskc, logits, labels[ : , num], b_preds)
                    all_preds[taskc].extend(c_preds)
                    all_labels[taskc].extend(c_labels)
                    temp[taskc] = c_preds
                else:
                    temp[taskb] = b_preds
                    
                    
            # add predictions to the input
            tc_hum_preds = torch.Tensor(temp['tc_hum']).to(device)
            tc_sar_preds = torch.Tensor(temp['tc_sar']).to(device)
            tc_off_preds = torch.Tensor(temp['tc_off']).to(device)
            tb_mot_preds = torch.Tensor(temp['tb_mot']).to(device)
            
            logits = torch.flatten(logits, 1)
            
            logits = torch.cat((logits, torch.unsqueeze(tc_hum_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_sar_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_off_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tb_mot_preds, 1)), dim=1)
            
            # predict for task A
            sen_preds, sen_labels = get_predictions_and_labels(model, 'ta_sen', logits, labels[ : , 0])
            all_preds['ta_sen'].extend(sen_preds)
            all_labels['ta_sen'].extend(sen_labels)
            
            
    # calculate, display, and save scores to a log file
    print("=======================================================")
    print("Validation Scores")
    print("=======================================================")
    results = calculate_results(memotion_number, all_preds, all_labels)
    save_results(memotion_number + "_v" + version, "val", results)
    
    return all_preds, all_labels


def calculate_results(memotion_number, all_preds, all_labels):
    results = {'ta_sen':0.0, 'tb_hum':0.0, 'tb_sar':0.0, 'tb_off':0.0, 'tb_mot':0.0, 'tc_hum':0.0, 'tc_sar':0.0, 'tc_off':0.0}
    
    if memotion_number == "memotion_1":
        avg_mode = "macro"
    else:
        avg_mode = "weighted"
        
    for key in results:
        results[key] = f1_score(all_labels[key], all_preds[key], average=avg_mode)
        
    return results


def save_results(memotion_folder, partition, results):
    # display results
    display.f1_scores(results)
    
    # save results
    f = open("./output/models/" + memotion_folder + "/" + partition + "_scores.txt", "w")
    
    f.write("[SCORES]")
    f.write("\nTask A: " + str(results['ta_sen']*100))
    f.write("\nTask B: " + str(((results['tb_hum'] + results['tb_sar'] + results['tb_off'] + results['tb_mot'])/4) *100))
    f.write("\nTask C: " + str(((results['tc_hum'] + results['tc_sar'] + results['tc_off'] + results['tb_mot'])/4) *100))
    
    f.write("\n[EXCEL]")
    f.write("\n" + str(results['ta_sen']*100))
    f.write(f"\n=AVERAGE({ results['tb_hum']*100 },{ results['tb_sar']*100 },{ results['tb_off']*100 },{ results['tb_mot']*100 })")
    f.write(f"\n=AVERAGE({ results['tc_hum']*100 },{ results['tc_sar']*100 },{ results['tc_off']*100 },{ results['tb_mot']*100 })")
    
    f.close()
    
    return


def test_model(memotion_number, model, memotion_dataloader):
    
    all_preds = {'ta_sen':[], 'tb_hum':[], 'tb_sar':[], 'tb_off':[], 'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
    
    # eval mode
    model_parts = ['base', 'ta_sen', 'tb_hum', 'tb_sar', 'tb_off', 'tb_mot', 'tc_hum', 'tc_sar', 'tc_off']
    model = mode_change(model, model_parts, "eval")
    
    with torch.no_grad():
        # for each batch in test set
        for data in memotion_dataloader.test_dataloader:
            
            data = data[0]
            data = data[ : , : num_keywords, : ]
            data = data.to(device)
            
            # forward pass to the model
            logits = model['base'](data)
            
            temp = {'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
            
            # predict for task B and C
            for taskb, taskc, num in zip(['tb_hum','tb_sar','tb_off','tb_mot'], ['tc_hum','tc_sar','tc_off', 'blank'], [1,2,3,4]):
                b_preds = get_predictions(model, taskb, logits, is_binary=True)
                all_preds[taskb].extend(b_preds)
                
                if num < 4:
                    c_preds = get_predictions_c(model, taskc, logits, b_preds)
                    all_preds[taskc].extend(c_preds)
                    temp[taskc] = c_preds
                else:
                    temp[taskb] = b_preds
                    
                    
            # add predictions to the input
            tc_hum_preds = torch.Tensor(temp['tc_hum']).to(device)
            tc_sar_preds = torch.Tensor(temp['tc_sar']).to(device)
            tc_off_preds = torch.Tensor(temp['tc_off']).to(device)
            tb_mot_preds = torch.Tensor(temp['tb_mot']).to(device)
            
            logits = torch.flatten(logits, 1)
            
            logits = torch.cat((logits, torch.unsqueeze(tc_hum_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_sar_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tc_off_preds, 1)), dim=1)
            logits = torch.cat((logits, torch.unsqueeze(tb_mot_preds, 1)), dim=1)
            
            # predict for task A
            sen_preds = get_predictions(model, 'ta_sen', logits)
            all_preds['ta_sen'].extend(sen_preds)
            
            
    # if memotion 1, adjust sentiment prediction
    if memotion_number == "memotion_1":
        all_preds['ta_sen'] = [x-1 for x in all_preds['ta_sen']]
        
    # save predictions
    save_test_predictions(memotion_number, all_preds)
    
    # calculate, display, and save scores to a log file
    evaluate_test_predictions(memotion_number, all_preds)
    
    # if memotion 1, adjust sentiment prediction
    if memotion_number == "memotion_1":
        all_preds['ta_sen'] = [x+1 for x in all_preds['ta_sen']]
        
    return all_preds


# save test predictions to answer.txt
def save_test_predictions(memotion_number, all_preds):
    # open test results file
    f = open("./output/predictions/" + memotion_number + "_v" + version + "/answer.txt", "w")
    
    samples_size = len(all_preds['ta_sen'])
    
    # for each task, store predictions
    for i in range(samples_size):
        f.write(str(all_preds['ta_sen'][i]))
        f.write("_")
        f.write(str(all_preds['tb_hum'][i]) + str(all_preds['tb_sar'][i]) + str(all_preds['tb_off'][i]) + str(all_preds['tb_mot'][i]))
        f.write("_")
        f.write(str(all_preds['tc_hum'][i]) + str(all_preds['tc_sar'][i]) + str(all_preds['tc_off'][i]) + str(all_preds['tb_mot'][i]))
        f.write("\n")
        
    f.close()
    
    return


def evaluate_test_predictions(memotion_number, all_preds):
    # get test labels
    all_labels = dataset.get_test_labels(memotion_number)
    
    print("=======================================================")
    print("Test Scores")
    print("=======================================================")
    results = calculate_results(memotion_number, all_preds, all_labels)
    save_results(memotion_number + "_v" + version, "test", results)
    
    return


# save the model to the destination
def save_model(model, model_num, memotion_version):
    # create a folder for the model parts
    output_directory = "./output/models/" + memotion_version + "/model_" + str(model_num + 1)
    os.makedirs(output_directory, exist_ok=True)
    
    save_dict = dict()
    
    # compose dictionary for all parts
    for part in model:
        save_dict[part] = model[part].state_dict()
        
    file_name = "model.pth"
    torch.save(save_dict, os.path.join(output_directory, file_name))
    
    return


# load the model
def load_model(model_num, memotion_version, memotion_dataset):
    output_directory = "./output/models/" + memotion_version + "/model_" + str(model_num + 1)
    
    # initialize model and load the parameters
    model = create_model(memotion_dataset)
    file_name = "model.pth"
    checkpoint = torch.load(os.path.join(output_directory, file_name))
    
    # store the parmeters to the new model
    for part in model:
        model[part].load_state_dict(checkpoint[part])
        
    return model


def transfer_output_files(memotion_number, model_num):
    # move logs to the subfolder
    source_dir = "./output/models/" + memotion_number + "_v" + version
    target_dir = "./output/models/" + memotion_number + "_v" + version + "/model_" + str(model_num + 1)
    
    file_names = os.listdir(source_dir)
    
    for file_name in file_names:
        if os.path.isfile(os.path.join(source_dir, file_name)):
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))
            
    return


# perform ensemble training, validation, and testing
def ensemble_train_val_test(memotion_number, memotion_dataset, num_models):
    
    if is_train:
        # create folder for output (models and graphs)
        os.makedirs("./output/models/" + memotion_number + "_v" + version, exist_ok=True)
        
        # create folder for predictions
        os.makedirs("./output/predictions/" + memotion_number + "_v" + version, exist_ok=True)
        
        # create a 2D list of random partition number
        set_numbers_list = dataset.make_partition_numbers(num_models)
        
        # for each models
        for model_num in range(num_models):
            print("Training for model #" + str(model_num + 1))
            
            # perform sampling with replacement (note: training set only)
            bootstrap_dataset = dataset.make_bootstrap_datasets(memotion_dataset, set_numbers_list[model_num])
            
            # create dataloader for the partitions
            print("Creating dataloader for the train, val, and test partitions...")
            memotion_dataloader = dataset.to_dataloader(bootstrap_dataset, batch_size)
            print("Done.")
            
            # create model
            model = create_model(bootstrap_dataset)
            
            # train emotion model (task B)
            print("=====================================================")
            print("Training Phase: Emotion")
            print("=====================================================")
            model = train_emotion(memotion_number, model, memotion_dataloader)
            
            # train intensity model (task C)
            print("=====================================================")
            print("Training Phase: Intensity")
            print("=====================================================")
            model = train_intensity(memotion_number, model, memotion_dataloader)
            
            # train sentiment model (task A)
            print("=====================================================")
            print("Training Phase: Sentiment")
            print("=====================================================")
            model = train_sentiment(memotion_number, model, memotion_dataloader)
            
            # save model
            save_model(model, model_num, memotion_number + "_v" + version)
            
            # store output to subfolder
            transfer_output_files(memotion_number, model_num)
            
            print("Training for model #" + str(model_num + 1) + " is complete.")
            
            
    # create validation and test prediction list
    val_preds_per_model = []
    test_preds_per_model = []
    val_labels = []
    
    # create dataloader for the partitions
    print("Creating dataloader for the train, val, and test partitions...")
    memotion_dataloader = dataset.to_dataloader(memotion_dataset, batch_size)
    print("Done.")
    
    # for each models
    for model_num in range(num_models):
        
        # load model
        model = load_model(model_num, memotion_number + "_v" + version, memotion_dataset)
        
        # perform validation and test + save information
        temp_preds, val_labels = validate_model(memotion_number, model, memotion_dataloader)
        val_preds_per_model.append(temp_preds)
        test_preds_per_model.append(test_model(memotion_number, model, memotion_dataloader))
        
        # store output to subfolder
        transfer_output_files(memotion_number, model_num)
        
        
    print("Performing majority voting...")
    # perform majority voting on validation predictions
    val_voting(memotion_number, val_preds_per_model, val_labels, num_models)
    
    # perform majority voting on test predictions
    test_voting(memotion_number, test_preds_per_model, num_models)
    print("Done.")
    
    
    return


# perform majority voting on validation predictions
def val_voting(memotion_number, val_preds_per_model, all_labels, num_models):
    all_preds = {'ta_sen':[], 'tb_hum':[], 'tb_sar':[], 'tb_off':[], 'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
    
    
    # for each task, get max voting
    for task in all_preds:
        
        temp_list = []
        for i in range(num_models):
            temp_list.append(val_preds_per_model[i][task])
            
        preds_per_task = np.asarray(temp_list)
        end_result, _ = stats.mode(preds_per_task)
        all_preds[task] = np.squeeze(end_result).tolist()
        
        
    # calculate, display, and save scores to a log file
    print("=======================================================")
    print("Validation Scores")
    print("=======================================================")
    results = calculate_results(memotion_number, all_preds, all_labels)
    save_results(memotion_number + "_v" + version, "val", results)
    
    
    return


# perform majority voting on test predictions
def test_voting(memotion_number, test_preds_per_model, num_models):
    all_preds = {'ta_sen':[], 'tb_hum':[], 'tb_sar':[], 'tb_off':[], 'tb_mot':[], 'tc_hum':[], 'tc_sar':[], 'tc_off':[]}
    
    
    # for each task, get max voting
    for task in all_preds:
        
        temp_list = []
        for i in range(num_models):
            temp_list.append(test_preds_per_model[i][task])
            
        preds_per_task = np.asarray(temp_list)
        end_result, _ = stats.mode(preds_per_task)
        all_preds[task] = np.squeeze(end_result).tolist()
        
        
    # if memotion 1, adjust sentiment prediction
    if memotion_number == "memotion_1":
        all_preds['ta_sen'] = [x-1 for x in all_preds['ta_sen']]
        
    # save predictions
    save_test_predictions(memotion_number, all_preds)
    
    # calculate, display, and save scores to a log file
    evaluate_test_predictions(memotion_number, all_preds)
    
    
    return


# load and preprocess the memotion dataset
def load_and_preprocess(memotion_number, memotion_dataset):
    
    X_train, y_train, X_val, y_val, X_test = [],[],[],[],[]
    
    # load dataset
    if memotion_number == "memotion_1":
        
        # open training and test set
        print("Loading dataset...")
        train_df = pd.read_csv("./input/dataset/" + memotion_number + "_train.csv")
        test_df = pd.read_csv("./input/dataset/" + memotion_number + "_test.csv")
        print("Done.")
        
        # store dataframes
        memotion_dataset.X_train = train_df
        memotion_dataset.X_test = test_df
        
        # label encoding
        print("Encoding the dataset...")
        memotion_dataset = dataset.label_encoding(memotion_number, memotion_dataset)
        print("Done.")
        
        # populate the dataset for the stratified split
        memotion_dataset.X_train = dataset.populate_train_val(memotion_dataset.X_train, tasks_list)
        
        # (X,y)
        X_train = memotion_dataset.X_train['response']
        y_train = memotion_dataset.X_train[tasks_list]
        X_test = memotion_dataset.X_test['response']
        
        # training and validation partitioning
        print("Creating stratified training and validation partition...")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train[tasks_list], random_state=42)
        print("Done.")
        
    else:
        
        # open training, validation, and test set
        print("Loading dataset...")
        train_df = pd.read_csv("./input/dataset/" + memotion_number + "_train.csv")
        val_df = pd.read_csv("./input/dataset/" + memotion_number + "_val.csv")
        test_df = pd.read_csv("./input/dataset/" + memotion_number + "_test.csv")
        print("Done.")
        
        # store dataframes
        memotion_dataset.X_train = train_df
        memotion_dataset.X_val = val_df
        memotion_dataset.X_test = test_df
        
        # label encoding
        print("Encoding the dataset...")
        memotion_dataset = dataset.label_encoding(memotion_number, memotion_dataset)
        print("Done.")
        
        # (X,y)
        X_train = memotion_dataset.X_train['response']
        y_train = memotion_dataset.X_train[tasks_list]
        X_val = memotion_dataset.X_val['response']
        y_val = memotion_dataset.X_val[tasks_list]
        X_test = memotion_dataset.X_test['response']
        
        
    if is_debug:
        display.value_counts(y_train, y_val, tasks_list)
        display.shape_and_type(X_train, y_train, X_val, y_val, X_test)
        
    # document and keyword extraction
    print("Extracting keyword and document embeddings...")
    print("> Training set")
    X_train = dataset.get_doc_key_embeddings(X_train, n_grams, num_keywords)
    print("> Validation set")
    X_val = dataset.get_doc_key_embeddings(X_val, n_grams, num_keywords)
    print("> Testing set")
    X_test = dataset.get_doc_key_embeddings(X_test, n_grams, num_keywords)
    print("Done with extracting keyword and document embeddings.")
    
    if is_debug:
        display.shape_and_type(X_train, y_train, X_val, y_val, X_test)
        
    # store preprocessed dataset
    memotion_dataset.X_train = X_train
    memotion_dataset.X_val = X_val
    memotion_dataset.X_test = X_test
    memotion_dataset.y_train = y_train
    memotion_dataset.y_val = y_val
    
    return memotion_dataset


# main program execution
if __name__ == '__main__':
    
    # start time counter
    print("Program started.")
    tic = time.perf_counter()
    
    # for each dataset
    for i in memotion_num_list:
        memotion_number = "memotion_" + str(i)
        print("=======================================================")
        print("[Dataset]")
        print(memotion_number)
        print("=======================================================")
        
        # load and preprocess the dataset
        print("[Dataset Loading and Preprocessing]")
        memotion_dataset = dataset.MemotionDataset()
        memotion_dataset = load_and_preprocess(memotion_number, memotion_dataset)
        print("=======================================================")
        
        if is_debug:
            display.shape_and_type(memotion_dataset.X_train,
                                   memotion_dataset.y_train,
                                   memotion_dataset.X_val,
                                   memotion_dataset.y_val,
                                   memotion_dataset.X_test)
        
        # ensemble training (multi-task)
        ensemble_train_val_test(memotion_number, memotion_dataset, n_ensemble)
        
        
    # end time counter
    toc = time.perf_counter()
    mins = (toc - tic) / 60
    print(f"\n\nProgram finished after {mins:0.2f} minutes.")

