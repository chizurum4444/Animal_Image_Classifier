# Animal Classifier

- I implemented a Deep CNN model to classify images of dogs and cats.

# Why Classify Cats and Dogs ?

- There could be many reasons why you would want a model to do this, but the reason stated here revolves around the utility for querying in search engines for pet breeds and what stores have them.

- Imagine wanting to know where you could find a particular type of cat or dog by a picture. You need a model that can identify them first. This model classifies an image as a dog or a cat.

- The aim is to challenge the fact of how hard it is for a computer to recognize something as simple as a dog vs a cat, something that human vision can do easily through the implementation of a PyTorch NN model.

# Data Collection

- Source: https://www.kaggle.com/competitions/dogs-vs-cats/data

- This dataset contains 25,000 images of cats and dogs split into the training dataset.

- The internet is filled with all kinds of images of real dogs and cats. The images in the dataset are all of different shapes and sizes, and all files given were in JPEG format.

- Inside the dataset, there are also captioned and edited photos that are taken into consideration.

# Pre-Processing

- The Dogs vs. Cats dataset included a training folder and a test folder, both full of images.

- Using the torchvision ImageFolder function was a quick and easy way to get the training data.

- I needed to re-format the train folder, splitting the images into cats and dogs folders. This step is required for ImageFolder to classify. Cats were labeled as 0, and dogs were labeled as 1.

- Upon import, I introduced the images as crops of 224x224 and converted them to grayscale.

# Neural Network Design

- I used a Deep CNN model to design this model.

- I used two convolutional layers with five kernels for the first one and three for the second, applying an activation layer after each convolution and max pool layer with a kernel of size 2.

# Training and Testing

- The data came in very large. For testing purposes, I made the training dataset smaller: 2000 of each dog and cat.

# How to Use

- Take a photo of a dog or a cat and place it in the test folder, let the test begin, and print out the model predictions.
- Input the file path of the image in the code.
- Run Testing/Make Prediction.
- Find the image in the test dataset and print the prediction.

Note: It would take a few steps for a beginner to use this without a front-end. I will update and correct any spelling errors.
