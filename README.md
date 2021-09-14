# PyTorch-AI-road-detection
Artificial intelligence model, training loop, concurrent model loader, inference<br/>
From now image processing runs concurrently

Custom AI model trained on custom dataset (1000 photos). <br/>
Learning rate = 0.0001<br/>
Optimizer = Adam<br/>
Results example from video.<br/>
Results could be better but due to low memory of GPU this is Maximum quality. 
![Road](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection/blob/main/Images/Example.gif)<br/>

# Model
AI model created with `PyTorch`.<br/>
The model is being trained on GPU, model's device is set to `"CUDA"` <br/>
<br/>
Model architecture
   - `16x Conv2d` layers with `ReLu` activation functions and `BatchNorm2d`
   - All of this in `nn.Sequential`
   - `Upsample` in decoder
   - `interpolate` Encoder_1


# The course of the loss function
100-120 epoch is enough for this model with this dataset. <br/>
For this project I used custom loss function - loss = ((y - y_pred) ** 2).mean()

![Loss](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection/blob/main/Images/Loss%20function.jpg)

# Dataset
Dataset consists of 1000 photos from rainy and sunny city. 50% rainy / 50% sunny photos<br/>
Link to dataset photos -
[Dataset](https://drive.google.com/drive/folders/1795opF54wK76r5snXs68OtGl2cR7g-3C?usp=sharing)<br/>
Example:<br/>
![Mask](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection/blob/main/Images/Image%20%26%20Mask.jpg)

# How to use 
In `Run_training.py` set folders with downloaded dataset like this <br/>

```python
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_sunny1/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_sunny2/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_rainy/")
folders_training.append("C:/Users/Samuel/PycharmProjects/Condapytorch/mestodataset2/City_rainy2/")
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
If you want to run inference with pretrained model - (you can choose)-
[Models](https://github.com/Samuel-Bachorik/PyTorch-AI-road-detection-classification/tree/main/Models)<br/>

In `segmentation_inference` set model Path <br/>
```python
# Path to trained weights
self.PATH = "./Model1"
```
In `Run_video_inference.py` set your desired video<br/>
```python
cap = cv2.VideoCapture("C:/Users/Samuel/PycharmProjects/Condapytorch/City.mp4")
```
Run `Run_video_inference.py`<br/>
When inference is done you will get output.avi video. <br/>

