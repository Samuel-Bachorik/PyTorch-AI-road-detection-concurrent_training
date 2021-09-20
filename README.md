# PyTorch-AI-road-detection
Artificial intelligence model, training loop, concurrent model loader, inference<br/>
From now image processing runs concurrently

Custom AI model trained on custom dataset (1000 photos). <br/>
Learning rate = 0.0001<br/>
Optimizer = Adam<br/>
GPU used = RTX 3060 12GB<br/>
Example:<br/>
![Road](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection-concurrent_training/blob/main/Images/Road_example.gif)<br/>

# Model
AI [Model](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection-concurrent_training/blob/main/Model_1.py) created with `PyTorch`.<br/>
The model is being trained on GPU, model's device is set to `"CUDA"` <br/>
<br/>
Model architecture
   - Encoder - decoder model
   - `16x Conv2d` layers divided in encoders and decoder
   - `ReLu` activatios +  `BatchNorm2d`
   -  All of this in `nn.Sequential`
   -  Kernel 3x3
   -  17 360 898 Model Parameters


# The course of the loss function
100-120 epoch is enough for this model with this dataset. <br/>
For this project I used mean squared error loss function  = ((y - y_pred) ** 2).mean()

![Loss](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection-concurrent_training/blob/main/Images/Loss%20function.jpg)

# Dataset
Dataset consists of 1300 photos from rainy and sunny city. 50% rainy / 50% sunny photos<br/>
Link to dataset photos -
[Dataset](https://drive.google.com/drive/folders/1aUeWMmBwkKbLvj19hiELGR9TBFUPIKsl?usp=sharing)<br/>
Example:<br/>
![Mask](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection/blob/main/Images/Image%20%26%20Mask.jpg)

# How to use 
In `Run_training.py` set folders with downloaded dataset like this <br/>

```python
folders_training.append("C:/Users/Samuel/PycharmProjects/Conda/City_dataset/City_sunny1/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Conda/City_dataset/City_sunny2/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Conda/City_dataset/City_rainy/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Conda/City_dataset/City_rainy2/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Conda/City_dataset/City_2/")
```
Then Run `Run_training.py` <br/>
<br/>
When model is trained you can run inference like this -<br/>
In `segmentation_inference` set saved model Path <br/>
```python
# Path to trained weights
self.PATH = "./Model1"
```
In `Run_video_inference.py` set your desired video<br/>
```python
cap = cv2.VideoCapture("C:/Users/Samuel/PycharmProjects/Condapytorch/City.mp4")
```
Run `Run_video_inference.py`<br/>
When inference is done you will get output.avi video. 

# Run with pretrained model
If you want to run inference with pretrained model you need to download model trained wieghts - <br/>
You can choose -
[Models weights](https://drive.google.com/drive/folders/11Cz2hnVdQutggVD7TjVIZHmEF48T4ErF?usp=sharing)<br/>

>IMPORTANT: Make sure your downloaded model weights corresponds with Model architecture.<br/>
>`Model_weights_1` for [Model 1](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection-concurrent_training/blob/main/Model_1.py)<br/>
>`Model_weights_2` for [Model 2](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection-concurrent_training/blob/main/Model_2.py) <br/>

In `segmentation_inference` set model Path <br/>

```python
# Path to trained weights
self.PATH = "Model_weights_1.pth"
```
In `Run_video_inference.py` set your desired video<br/>
```python
cap = cv2.VideoCapture("C:/Users/Samuel/PycharmProjects/Condapytorch/City.mp4")
```
Run `Run_video_inference.py`<br/>
When inference is done you will get output.avi video. <br/>


# Test video
For best result use video with aspect ratio `4:3` and ressolution `1024x768` <br/>
You can test model on your own video or you can download here - [Download Video](https://drive.google.com/file/d/13RuSzPdqhz8a-k9XH5ni0hIREOTa3v9V/view?usp=sharing)<br/>
