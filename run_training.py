from process_dataset import ProcessDataset
from Model import Model
import torch
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Load training images
    folders_training = []
    folders_training.append("C:/Users/Samuel/PycharmProjects/pythonProject/City_dataset/City_sunny1/")
    folders_training.append("C:/Users/Samuel/PycharmProjects/pythonProject/City_dataset/City_sunny2/")
    folders_training.append("C:/Users/Samuel/PycharmProjects/pythonProject/City_dataset/City_rainy/")
    folders_training.append("C:/Users/Samuel/PycharmProjects/pythonProject/City_dataset/City_rainy2/")

    # Asign classes
    classes_ids = [8, 12]
    classes_count = len(classes_ids)

    model = Model() # Load model

    # Time estimating variables
    epochminus, arrayloss, arrayepoch, lossforavg = 0, [], [], 0

    # Print start time
    print(time.time())
    # At least 150 epoch for great results
    epochcount = 20
    # Batch size, (You can increase this number by x2 if you have enough memory on GPU) (32,64..)
    batch_size = (64)

    # Load images (height and width must be divisible by 32)
    dataset = ProcessDataset(folders_training, folders_training, classes_ids, height=384, width=512, augmentation_count=10)

    # Training loop
    for epoch in range(epochcount):
        # Time estimating variables
        epochminus += 1
        timestart = time.time()

        # calculate batch_count
        batch_count = (dataset.get_training_count() + batch_size) // batch_size
        print(batch_count, "Batch_count")

        # Set optimizer for our model parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Print current epoch
        print("EPOCH - ", epoch)

        # Batch loop
        for batch in range(batch_count):
            # Get batch from model loader
            x, y = dataset.get_training_batch(batch_size)

            # Put images on GPU
            x = x.to(model.device)
            y = y.to(model.device)

            # Push images to model
            y_pred = model.forward(x)

            # Calculate loss for optimizer
            loss = ((y - y_pred) ** 2).mean()

            # Get loss number for graph
            lossforavg += float(loss.data.cpu().numpy())

            # Reset gradients ,Find gradient,Update x
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Graphing variables
        arrayepoch.append(epoch)
        arrayloss.append(lossforavg / batch_count)
        lossforavg = 0

        # Time estimating variables
        timetoend = (epochcount - epochminus) * (time.time() - timestart)
        dt_object = datetime.fromtimestamp(timetoend + time.time())
        print(dt_object, "time to end")

        # save model weights every 10th epoch
        if epoch % 10 == 0:
            PATH = './Model_weights.pth'
            torch.save(model.state_dict(), PATH)

    # Save final model
    PATH = './Trained_model_weights.pth'
    torch.save(model.state_dict(), PATH)

    # Show and save loss function graph
    plt.plot(arrayepoch, arrayloss)
    plt.savefig('loss.png')
    plt.show()
