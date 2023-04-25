import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch.nn.functional as F

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.image_list[index])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.image_list[index].split('_')[0]) 
        label = torch.tensor(label)
        #print(image.shape)
        return image, label

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(12000, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


train_transforms = transforms.Compose([
    transforms.Resize((62, 100)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # grayscale
    transforms.Grayscale(num_output_channels=1)
])

# Load the training dataset
train_dataset = CustomDataset('../data/Datasets_digits', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model and the optimizer
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
# Train the model
num_epochs = 32
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = images  # batch x
            b_y = labels   # batch y
            output = cnn(b_x)[0] 
            #print(output.shape), print(b_y.shape,b_x.shape)              
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            if (i+1) % 32 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    torch.save(cnn, "digitalreadingmodel.pt")
    print("saved")


if __name__ == '__main__':

    train(num_epochs, model, train_loader)

    print('Finished Training')
