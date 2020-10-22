# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#################################################################################################

import time

start_time = time.time()

# Importing numpy library for math computation
import numpy as np

# Importing pytorch library for deep learning
import torch as py

#importing matplotlib for data visualization functionality
import matplotlib.pyplot as plt

#importing style for visualization style
from matplotlib import style

#importing neural network object in pytorch library
import torch.nn as nn

#importing neural network object in pytorch library
import torch.nn.functional as F

#importing data object in pytorch library
import torch.utils.data as utils

#importing transforms object from torchvision library
import torchvision.transforms as transforms

#importing a learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau



#################################################################################################
# Defining all classes and functions beforehand
#################################################################################################

style.use('fivethirtyeight')

#################################################################################################

# Define function to convert a 28*28 array into a 28*28 tensor for image visualization purposes

def array2d(x):
            
    plt.figure()
    plt.imshow(x.reshape(28,28))
    plt.show()
    
    
    
def imagetransform(x, transform):
    
    y = np.zeros(784)
    
    for i in range(x.shape[0]):
        
        y = transform(x[i,:].reshape(28,28).astype(int)).view(784).numpy()
        
#         for j in range(x.shape[1]): 
            
#             x[i,j] = int(y[j])
            
    return x



# Defining neural network models
    
# Neural network with one hidden layer

class NN(nn.Module):
    
    def __init__(self, D_in, H, D_out):
        super(NN, self).__init__()
        
        self.cnn = nn.Sequential(
                
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
    
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 3136)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Neural network with two hidden layers

class NN2(nn.Module):
    
    def __init__(self, D_in, H1, H2, D_out):
        super(NN2, self).__init__()
        
        self.cnn = nn.Sequential(
                
                nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        
        self.linear1 = nn.Linear(int(D_in/4), H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,1568)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))        
        x = self.linear3(x)
        return x
    
# Neural network with three hidden layers

class NN3(nn.Module):
    
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(NN3, self).__init__()
        
        self.cnn = nn.Sequential(
                
                nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4,16,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        
        self.linear1 = nn.Linear(int(D_in/4), H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 1568)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))      
        x = F.relu(self.linear3(x))        
        x = self.linear4(x)
        return x



# Defining the training function

def train(model, trainloader, criterion, optimizer, epochs = 2):
    
    diagnostics = []
    
    for epoch in range(epochs):
        
        print ("Training at epoch - ", epoch )

        i = 0
        
        for i, (x,y) in enumerate(trainloader):
            
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            diagnostics.append(loss.data.item())

    return diagnostics



# Defining the validation function

def validate(model, validationloader):
    
    print("Validating")
    
    diagnostics = []
        
    for i, (x,y) in enumerate(validationloader):
                
        yhat = model(x)
        _, label=py.max(yhat,1)
        
        if (label==y):
            diagnostics.append(1)
            
        else:
            diagnostics.append(0)
            
#            if i  < 2100:
#                array2d(x)
#                print("This was miscategorized as ", label.item())
    
    return diagnostics


# Defining the validation function

def test(model, testloader):
    
    print("Testing started")
    
    f = open("output.txt","w+")
    
    f.write("ImageId,Label\r\n")
        
    for i, (x,y) in enumerate(testloader):
                
        yhat = model(x)
        _, label=py.max(yhat,1)
        
        f.write(str(i+1) + "," + str(label.item()) + "\r\n")
        
        print("Testing file #", i)
              
    f.close()

    print("Testing complete")



#################################################################################################
# Defining the main function
#################################################################################################
    
def main_(dataload_ = True, dataexplore_ = False, normalizeimage_ = True, transformimage_ = True, train_ = True, validate_ = True, test_ = False):
    
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomAffine(20),
                                    transforms.ToTensor()])
    
    training_results = []
    validation_results = []

    # Loading raw data into a numpy arrays
    
    if (dataload_):
    
        try: 
            
            data = np.loadtxt(fname = "/kaggle/input/digit-recognizer/train.csv", delimiter = ",", skiprows=1)
            mnist_labels = data[:,0]
            mnist_data = data[:,1:785]
            
            print("Train data loaded successfully")
            
            test_data = np.loadtxt(fname = "/kaggle/input/digit-recognizer/test.csv", delimiter = ",", skiprows=1)    
            
            print("Test data loaded successfully")
            
        except:
            
            print("Could not load data")
            
            
            
    if (normalizeimage_):
        
        mnist_data /= 255.0

        test_data /= 255.0
        
        
    
    if (transformimage_):
        
        mnist_data = imagetransform(mnist_data, transform)
    
    
    
    # Exploring the transformed images
    
    if (dataexplore_):
        
        images_ = 10
        
        for _ in range(images_):
            
            array2d(mnist_data[_])
            

    
    # Training the model
    
    if (train_):
        
        input_layer    = 3136
        hidden_layer_1 = 100
        hidden_layer_2 = 50
        hidden_layer_3 = 25
        output_layer   = 10
        
        model_1 = NN(input_layer, hidden_layer_1, output_layer)
        model_2 = NN2(input_layer, hidden_layer_1, hidden_layer_2, output_layer)
        model_3 = NN3(input_layer, hidden_layer_1, hidden_layer_2, hidden_layer_3, output_layer)
        
        mnist_labels = py.from_numpy(mnist_labels).type(py.LongTensor)
        mnist_data = py.from_numpy(mnist_data).view(42000,1,28,28).type(py.FloatTensor)
                
        trainset = utils.TensorDataset(mnist_data[0:42000,:,:,:], mnist_labels[0:42000])
        trainloader = utils.DataLoader(trainset, batch_size = 300, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.01
        optimizer = py.optim.SGD(model_1.parameters(), lr = learning_rate)
                    
        training_results = train(model_1, trainloader, criterion, optimizer, epochs = 80)
        
        
    # Validating the model
        
    if (validate_):
        
        validationset = utils.TensorDataset(mnist_data[31501:42000,:,:,:], mnist_labels[31501:42000])
        validationloader = utils.DataLoader(validationset, batch_size = 1)
        
        validation_results = validate(model_1, validationloader)
        
    if (test_):

        test_data = py.from_numpy(test_data).view(28000,1,28,28).type(py.FloatTensor)
        test_labels = py.zeros(28000).type(py.LongTensor)
        
        testset = utils.TensorDataset(test_data, test_labels)
        testloader = utils.DataLoader(testset, batch_size = 1)
        
        test(model_1, testloader)
        
        
        
    return training_results, validation_results


#################################################################################################
# Calling the main function
#################################################################################################


training_results, validation_results = main_(dataload_ = True, dataexplore_ = False, normalizeimage_ = False, transformimage_ = True, train_ = False, validate_ =False, test_ = False)


# Evaluating the model

fig = plt.figure(1)
ax1 = fig.add_subplot(1,1,1)         
ax1.plot(training_results)
plt.yscale('log')
#plt.xlim(left = len(training_results) - 1000, right = len(training_results))
plt.show()

print("Training loss =", training_results[-1])
#print("Validation_accuracy =", round(sum(validation_results)/len(validation_results)*100,0), "%")

print("Time elapsed = ", time.time() - start_time)
