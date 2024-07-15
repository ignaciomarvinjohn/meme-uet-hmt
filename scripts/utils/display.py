import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def value_counts(y_train, y_val, tasks_list):
    print("------------------------------")
    print("**[Value counts]**")
    for task in tasks_list:
        print("------------------------------")
        print(">>" + task + "<<")
        print("--[Train]--")
        print(y_train[task].value_counts())
        print("--[Val]--")
        print(y_val[task].value_counts())
    print("------------------------------")
    
    return


def shape_and_type(X_train, y_train, X_val, y_val, X_test):
    print("------------------------------")
    print("**[Shape]**")
    print("------------------------------")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("------------------------------")
    print("**[Type]**")
    print("------------------------------")
    print("X_train type:", type(X_train))
    print("y_train type:", type(y_train))
    print("X_val type:", type(X_val))
    print("y_val type:", type(y_val))
    print("X_test type:", type(X_test))
    print("------------------------------")
    
    return


def loss_per_epoch(epoch, data_size, train_loss_history, valid_loss_history, tasks_list):
    print("------------------------------")
    print("Epoch:", epoch+1)
    
    # for each task
    for task in tasks_list:
        
        # get the latest recorded loss for that task
        train_loss = train_loss_history[task][-1]
        valid_loss = valid_loss_history[task][-1]
        
        # print training and validation loss
        print(f"{task.capitalize()} \t\t Training Loss: {train_loss / data_size['train']} \t\t Validation Loss: {valid_loss / data_size['val']}")
        
    print("------------------------------")
    
    return


def make_loss_plot(memotion_folder, task_type, train_loss_history, valid_loss_history, tasks_list):
    
    # for each task
    for task in tasks_list:
        
        # create visualization of the loss
        history_length = len(train_loss_history[task])
        x_axis = range(1, history_length+1)
        
        plt.plot(x_axis, train_loss_history[task], label='Training Loss')
        plt.plot(x_axis, valid_loss_history[task], label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.xticks(np.arange(0, history_length+1, int(history_length / 10)))
        
        plt.legend(loc='upper right')
        plt.savefig('./output/models/' + memotion_folder + '/' + task_type + "_" + task + '.png')
        plt.close()
        
    return


def f1_scores(results):
    print("\n================================[TASK A]================================\n")
    print("F1 score:", results['ta_sen']*100)
    print("\n================================[TASK B]================================\n")
    print("==========    humour    ==========")
    print("F1 score:", results['tb_hum']*100)
    print("==========  sarcastic   ==========")
    print("F1 score:", results['tb_sar']*100)
    print("==========  offensive   ==========")
    print("F1 score:", results['tb_off']*100)
    print("========== motivational ==========")
    print("F1 score:", results['tb_mot']*100)
    print("================================================")
    print("Average F1 score:", ((results['tb_hum'] + results['tb_sar'] + results['tb_off'] + results['tb_mot'])/4) *100)
    print("\n================================[TASK C]================================\n")
    print("==========    humour    ==========")
    print("F1 score:", results['tc_hum']*100)
    print("==========  sarcastic   ==========")
    print("F1 score:", results['tc_sar']*100)
    print("==========  offensive   ==========")
    print("F1 score:", results['tc_off']*100)
    print("========== motivational ==========")
    print("F1 score:", results['tb_mot']*100)
    print("================================================")
    print("Average F1 score:", ((results['tc_hum'] + results['tc_sar'] + results['tc_off'] + results['tb_mot'])/4) *100)
    
    return

