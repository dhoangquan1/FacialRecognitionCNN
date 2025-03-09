# Facial Expression Recognition CNN (FER)
*This project is a demo for educational purposes as part of course*

This program is trained on the FER-2013 dataset, and uses Convolutional Neural Network (CNN) to analyze and process visual data. 

FER-2013 is a dataset of 7-category huaman facial expressions that is used to test and train AI for facial expression recognition. FER-2013 contains 28,709 data examples for training, and 3,589 data examples for testing.

CNN is a deep learning architecture that often used to process grid-like data, such as images. CNN can automatically adjust and apply filters (also known as kernels) to images to extract the most useful features of images, such as edges, shapes, or textures. The filters are used to recognize complex patterns within the 7 human expressions: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

------------------------
## ⚙️ Technologies Used
-----------------------
[![My Skills](https://skillicons.dev/icons?i=py, pytorch,sklearn)](https://skillicons.dev)

------------------------
## 🧮 Attributes
-----------------------
These are attributes that you can change to alter how the AI learns.

### Pre-Processing Data

*These attributes dictate the image transformation, which can make the AI more robust and creates more data for the AI to learn*

| Name | Description | Value |
|------|-------------|---------|
| Resize | Resize the dimension of the input | `48x48` |
| RandomPad | Randomly Pad the image | `padding=4, padding_mode=‘reflect’`  |
| RandomResizedCrop | Randomly resize and crop the inputs | `size=(48,48), scale=(0.5, 1.0)` |
| RandomHorizontalFlip | Randomly flip the inputs horizontally | `50% chance` |
| Normalize | Normalize the inputs to avoid extreme image transformation | `mean=[0.5], std=[0.5]` |

### Hyperparameters

*These attributes dictate how fast and efficient the AI will learn per epoch, change based on if the AI is train on CPU or GPU*

| Name | Description | Value |
|------|-------------|---------|
| Batch Size | Number of training samples that are processed together | `128` |
| N-Epochs | Number of iterations that the AI will train | `30`  |
| Optimizer | The algorithm that adjusts the network's parameters to reduce error | `Adam(model.parameters(), lr=0.001, weight_decay=1e-4)` |
| Scheduler | The algorithm that adjusts the optimizer to encourage efficient changes in learning rate | `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)` |

------------------------
## 🧠 Model Architecture
-----------------------
The Model uses a 4-layer convolution design with 2 linear layers, which converges to the 7 expresion.
It also split the dataset to 80% used for training, and 20% used for validating.
These layers dictate the quality of training, which can enhance the AI accuracy.

#### Layer Definitions

*Values such as x,y change according to layer depths*

| Layer | Process | Value |
|------|-------------|---------|
| Conv2D | Apply filters to inputs and extract features | `x, y, kernel_size=3, padding=1` |
| BatchNorm | Normalize the inputs of layer to increase efficiency | `y` |
| ReLU | Activation function we chose |  |
| MaxPool2D | Reduce size of image while retaining important feature | `kernel_size=2, stride=2` |
| Dropout | Randomly drop neurons during training, to avoid overfitting | `p=0.3-0.5` |
| Flatten | Compress all filters from convolution to 1D |  |
| Linear | Transform direct ouputs to inputs | `x, y` |

#### Layer Forwarding

| Layer | Process | Neurons (Pre-MaxPool) |
|------|-------------|---------|
| Convolution-1 | Conv2D(1,32) > BatchNorm > ReLU > MaxPool2D > Dropout(0.5) | `18432` |
| Convolution-2 | Conv2D(32,64) > BatchNorm > ReLU > MaxPool2D > Dropout(0.3) | `9216` |
| Convolution-3 | Conv2D(64,128) > BatchNorm > ReLU > MaxPool2D > Dropout(0.5) | `4608` |
| Convolution-4 | Conv2D(128,256) > BatchNorm > ReLU > MaxPool2D > Dropout(0.3) | `2304` |
| Flatten |  | `2304` |
| Linear-1 | Linear(2304,256) > BatchNorm > ReLU > Dropout(0.5) | `256` |
| Linear-2 | Linear(256,7) | `7` |

------------------------
## 🚀 Getting Started
-----------------------
### Requirements:
- Pytorch verion compatible to your machine
- Python 3.9+ (recommend 3.12+)

### Set up dependencies:
- Create a python environment:
    ```
    python -m venv venv
    ```
- Activate the python environment:
    ```
    .\venv\Scripts\activate
    ```
- Download dependencies:
    ```
    pip install -r requirements.txt
    ```

### Get datasets:
- Get both training and testing datasets here:
[FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)

------------------------
## 🖥️ Running the program
-----------------------

### To train the model:
- Start training with the following command:
    ```
    python code_1.py
    ```

### Program outputs:
- **model_best**: a copy of your best model from the training
- **epoch_accs**: the accuracies of the model from each epoch training and validation
- **images**: the performance graph, confusion matrix, and 3 images that were misclassified by the best model.

------------------------
## 🎯 Best Model Results
-----------------------
### Overall Accuracy
- Our best model has accuracy of **60%**

### Performance Graph
<img src="/documents/Performance.png" style="max-width: 100%; height: auto;"/>

### Confusion Matrix
<img src="/documents/Confusion.png" style="max-width: 100%; height: auto;" />

### Misclassified Images
<img src="/documents/Misclass.png" style="max-width: 100%; height: auto;" />

------------------------
## 📚 Documentations
-----------------------
For more information on the AI developemnt process and design, check out the documentations in /documents to see what experiments we did to achieve the current best model.

------------------------
## 🙏 Acknowledgements
-----------------------
- Professor Ruiz - CS4341: Introduction to AI
- Contributors/Team Members 
    <div style="display: flex; gap: 10px; margin-top: 10px;">
        <a href="https://github.com/dhoangquan1">
            <img src="https://github.com/dhoangquan1.png" width="50" style="border-radius: 25px; overflow: hidden;">
        </a>
        <a href="https://github.com/ElijahWPI">
            <img src="https://github.com/ElijahWPI.png" width="50" style="border-radius: 25px; overflow: hidden;">
        </a>
    </div>