import torch
import torch.nn as nn
import torch.nn.functional as F


#==========================================================================================
# Main UET model class
class UET(nn.Module):
    # Declaring the Architecture
    def __init__(self, key_emb_sizes):
        super(UET, self).__init__()
        
        self.emb_1 = key_emb_sizes[0]
        self.emb_2 = key_emb_sizes[1]
        self.emb_3 = key_emb_sizes[2]
        self.emb_4 = key_emb_sizes[3]

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_1, out_channels=self.emb_2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.emb_2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_2, out_channels=self.emb_3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.emb_3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_3, out_channels=self.emb_4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.emb_4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2)
        )
        
        # Transformer
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.emb_4, nhead=8, batch_first=True), num_layers=1)
        
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_4, out_channels=self.emb_3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.emb_3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_3, out_channels=self.emb_2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.emb_2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_2, out_channels=self.emb_1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.emb_1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.2)
        )
        
    # Forward Pass
    def forward(self, x):
        
        x = torch.transpose(x, 1, 2) # B x N x W -> B x W x N
        
        # encoder layer
        x1 = x
        x = self.enc1(x)
        x2 = x
        x = self.enc2(x)
        x3 = x
        x = self.enc3(x)
        
        # transformer layer
        x = torch.transpose(x, 1, 2) # B x W x N -> B x N x W
        x = self.transformer(x)
        x = torch.transpose(x, 1, 2) # B x N x W -> B x W x N
        
        # decoder layer
        x = self.dec1(x) + x3
        x = self.dec2(x) + x2
        x = self.dec3(x) + x1
        
        x = torch.transpose(x, 1, 2) # B x W x N -> B x N x W
        
        return x


#==========================================================================================
# Blank (No Feature Extraction)
class PassLayer(nn.Module):
    # Declaring the Architecture
    def __init__(self):
        super(PassLayer, self).__init__()
        
        return
        
    # Forward Pass
    def forward(self, x):
        
        return x


#==========================================================================================
# Task A Prediction Layer (Hierarchical)
class TaskAPredictLayerH(nn.Module):
    def __init__(self, emb_size, in_size, extra, out_size):
        super(TaskAPredictLayerH, self).__init__()
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(emb_size + extra, in_size[0]),
            nn.BatchNorm1d(in_size[0]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            
            nn.Linear(in_size[0], in_size[1]),
            nn.BatchNorm1d(in_size[1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            
            nn.Linear(in_size[1], out_size)
        )
        
        return
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        
        return x


#==========================================================================================
# Task B Prediction Layer (Hierarchical)
class TaskBPredictLayerH(nn.Module):
    def __init__(self, emb_size, in_size, out_size):
        super(TaskBPredictLayerH, self).__init__()
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, in_size[0]),
            nn.BatchNorm1d(in_size[0]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            
            nn.Linear(in_size[0], in_size[1]),
            nn.BatchNorm1d(in_size[1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            
            nn.Linear(in_size[1], out_size)
        )
        
        return
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        
        return x


#==========================================================================================
# Task C Prediction Layer (Hierarchical)
class TaskCPredictLayerH(nn.Module):
    def __init__(self, emb_size, in_size, out_size):
        super(TaskCPredictLayerH, self).__init__()
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, in_size[0]),
            nn.BatchNorm1d(in_size[0]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            
            nn.Linear(in_size[0], in_size[1]),
            nn.BatchNorm1d(in_size[1]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            
            nn.Linear(in_size[1], out_size)
        )
        
        return
        
    def forward(self, x, bin_pred):
        
        x = torch.flatten(x, 1)
        x = self.mlp(x) # B x 4
        
        for b in range(x.shape[0]):
            if bin_pred[b] == 0:
                x[b] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
                
        return x

