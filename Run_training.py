from Model_loader import DatasetSegmentation
from Model import Model
import torch.nn as nn
import torch
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Function for weight init
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        print('initializing conv2d weight')
        torch.nn.init.xavier_uniform_(m.weight)

#Load training images
folders_training = []
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_sunny1/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_sunny2/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_rainy/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_rainy2/")

#Asign classes
classes_ids = [8, 12]
classes_count = len(classes_ids)

#Load images (height and width must be divisible by 32)
dataset = DatasetSegmentation(folders_training, folders_training, classes_ids, height=480, width=640)

#Load model
model = Model()

#Time estimating variables
epochminus = 0
arrayloss = []
arrayepoch = []
lossforavg = 0

#Print start time
print(time.time())

# At least 150 epoch for great results
epochcount = 150
#Batch size, (You can increase this number by x2 if you have enough memory on GPU) (32,64..)
batch_size = (16)

#Training loop
for epoch in range(epochcount):
    # Time estimating variables
    epochminus += 1
    timestart = time.time()

    #calculate batch_count
    batch_count = (dataset.get_training_count() + batch_size) // batch_size
    print(batch_count, "batch_count")

    # Set optimizer for our model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Print current epoch
    print("EPOCH - ",epoch)

    # Batch loop
    for batch in range(batch_count):

        # Get batch from model loader
        x, y = dataset.get_training_batch(16)

        # Put images on GPU
        x = x.to(model.device)
        y = y.to(model.device)

        #Push images to model
        y_pred = model.forward(x)

        #Calculate loss for optimizer
        loss = ((y - y_pred) ** 2).mean()

        # Get loss number for graph
        lossnumber = float(loss.data.cpu().numpy())
        lossforavg += lossnumber

        #Reset gradients
        optimizer.zero_grad()
        #Find gradient
        loss.backward()
        #Update x
        optimizer.step()

    # Graphing variables
    arrayepoch.append(epoch)
    lossavg  = lossforavg/batch_count
    arrayloss.append(lossavg)
    lossforavg = 0

    # Time estimating variables
    timeend = time.time()
    epoch1time = timeend - timestart
    timetoend = epochcount - epochminus
    timetoend = timetoend * epoch1time
    dt_object = datetime.fromtimestamp(timetoend + time.time())
    print(dt_object, "time to end")

    # save model weights every 10th epoch
    if epoch % 10 == 0:
        PATH = './Model_weights.pth'
        torch.save(model.state_dict(), PATH)


#Save final model
PATH = './Trained_model_weights.pth'
torch.save(model.state_dict(), PATH)

# Show and save loss function graph
plt.plot(arrayepoch, arrayloss)
plt.savefig('loss.png')
plt.show()
