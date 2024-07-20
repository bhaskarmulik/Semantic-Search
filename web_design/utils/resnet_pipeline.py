import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
import os
import numpy as np
from PIL import Image


class ImageVec(nn.Module, transforms):

    '''
    This class is used to get the image embeddings from the ResNet50 model
    For this, we first load the model and then remove the last layer of the model
    We then transform the image according to the model's requirements for optimal results
    Lastly, we get the feature vectors from the model.
    They are obtained in shape(num_of_images,2048)
    '''

    def __init__(self):
        super().__init__()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.model= resnet50(weights = ResNet50_Weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.embed = self.obtain_child(self.model)

    def obtain_child(self, model):
        return nn.Sequential(*list(self.model.children())[:-1])

    def get_vec(self, img):
        img = Image.open(img)   #Open the image
        img = self.transforms(img).unsqueeze(0) #Apply the transforms and unsqueeze to add a batch dimension
        img = img.to(self.device)   #Move the image to the device
        return self.embed(img)  #Get the embeddings