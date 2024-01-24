# firejet-intern
firejet assessment

## Model Details
- ImageNet weights
- InceptionV3 Architecture

## General Approach
- Create train, test splits (80/20)
- Resized images to 256x256
- Create one-hot encoded label set
- Transfer learning with pre-trained ImageNet model, freeze layers except top

## Model Performance
- Model accuracy of 0.7657
![image](assets/Screenshot%202024-01-24%20at%207.13.43 PM.png)