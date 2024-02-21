import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DeiTModel, DeiTConfig
from utils import stochastic_dropout


class CustomResNet(nn.Module):
    def __init__(self, num_class):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer 
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    def forward(self, x):
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.fc(x)
        return x


class Custom_ViT(nn.Module):
    def __init__(self, num_class):
        super(Custom_ViT, self).__init__()
        self.num_class = num_class

        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        self.ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        x2 = self.ViT_fc(x2)
        return x2
    






class Multi_head_MLP(nn.Module):
    def __init__(self, num_class, num_heads):
        super(Multi_head_MLP, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
       
        # Create multiple heads
        self.heads = nn.ModuleList()

        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads.append(head)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Forward pass through multiple heads
        List = []
        for head in self.heads:
            output = head(x)
            List.append(output)
        
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        return h, head_outputs
    






class Multi_head_CNN_MLP_1(nn.Module):
    def __init__(self, num_class, num_heads):
        super(Multi_head_CNN_MLP_1, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Create multiple heads
        self.head_CNN = nn.ModuleList()
        self.head_MLP = nn.ModuleList()

        for _ in range(num_heads):
            CNN = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
            self.head_CNN.append(CNN)

            MLP = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.head_MLP.append(MLP)

    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Forward pass through multiple heads
        List = []
        for cnn_module, mlp_module in zip(self.head_CNN, self.head_MLP):
            output = cnn_module(x)
            output = torch.flatten(output, 1)
            output = mlp_module(output)
            List.append(output)

        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        return h, head_outputs
    








class Multi_head_CNN_MLP_2(nn.Module):
    def __init__(self, num_class, num_heads):
        super(Multi_head_CNN_MLP_2, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Create multiple heads
        
        self.head_CNN = nn.ModuleList()
        self.head_MLP = nn.ModuleList()

        for _ in range(num_heads):
            CNN = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
            self.head_CNN.append(CNN)

            MLP = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.head_MLP.append(MLP)


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Forward pass through ResNet base
        x = self.resnet(x)

        # Forward pass through multiple heads 
        List = []
        for cnn_module, mlp_module in zip(self.head_CNN, self.head_MLP):
            output = cnn_module(x)
            output = torch.flatten(output, 1)
            output = mlp_module(output)
            List.append(output)

        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)
        return h, head_outputs
    





class EnsembleModel_CNN_ViT_1(nn.Module):
    def __init__(self, num_class):
        super(EnsembleModel_CNN_ViT_1, self).__init__()
        self.num_class = num_class
        # Define ResNet50 model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.CNN = nn.Sequential(*modules)

        self.CNN_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
        
        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        self.ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
        

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        
        x1 = self.CNN(x)
        x1 = torch.flatten(x1, 1)
        x1 = self.CNN_fc(x1)
        
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        x2 = self.ViT_fc(x2)

        List = [x1, x2]
        head_outputs = torch.cat(List, dim=1)
        h = head_outputs.view(batch_size, 2, self.num_class)
        
        return h, head_outputs
    






class EnsembleModel_CNN_ViT_2(nn.Module):
    def __init__(self, num_class, num_heads):
        super(EnsembleModel_CNN_ViT_2, self).__init__()
        self.num_class = num_class
        self.num_heads = num_heads
        # Define ResNet50 model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.CNN = nn.Sequential(*modules)
 
        
        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        # Create multiple heads for CNN
        self.heads_CNN = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            CNN_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_CNN.append(CNN_fc)

        
        # Create multiple heads for ViT
        self.heads_ViT = nn.ModuleList()
        for _ in range(int(num_heads/2)):
            ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )
            self.heads_ViT.append(ViT_fc)
            
    
        

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        
        x1 = self.CNN(x)
        x1 = torch.flatten(x1, 1)
        List_CNN = []
        for head_CNN in self.heads_CNN:
            output = head_CNN(x1)
            List_CNN.append(output)

        head_CNN = torch.cat(List_CNN, dim=1)
            

        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        List_ViT = []
        for head_ViT in self.heads_ViT:
            output = head_ViT(x2)
            List_ViT.append(output)

        head_ViT = torch.cat(List_ViT, dim=1)

        head_outputs = torch.cat((head_CNN, head_ViT), dim=1)
        h = head_outputs.view(batch_size, self.num_heads, self.num_class)

        return h, head_outputs






        

        
            