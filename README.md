# Image-Text Matching using CLIP

This project implements an image-text matching model using the CLIP (Contrastive Language-Image Pretraining) framework. The model is capable of encoding both images and textual descriptions into fixed-size vectors and comparing them to find relevant matches between images and text.

## Trained Weights Link
Add this in  
Make a new folder with the name of content
`content/best.pt`

Link: [Trained Weights Link](https://drive.google.com/file/d/13s0pAYkoIKWPjf4EBeetNNofTJ7edapw/view?usp=sharing)

## Overview

The project consists of the following components:

1. `Dataset:` The dataset class prepares the data for training by encoding textual descriptions using a DistilBERT tokenizer and loading and preprocessing images.

2. `Image Encoder:` This component encodes images into fixed-size vectors using a pre-trained ResNet50 model.

3. `Text Encoder:` Textual descriptions are encoded into fixed-size vectors using a DistilBERT model.

4. `Projection Head:` The projection head projects both image and text embeddings into a shared space for comparison.

5. `CLIP Model:` This component utilizes the above modules to implement the CLIP model. It computes the loss function based on the dot product similarity between image and text embeddings.

6. `Inference:` After training, the model can perform inference by retrieving relevant images based on a given text query.

7. `Tokenizer:` Pretrained Bert Tokenizer is used for the tokenization. 

## Installation

To install the dependencies, run:

`pip install -r requirements.txt`


## Usage

1. Prepare your dataset and ensure it is formatted appropriately.
2. Create a tokenizer object using the HuggingFace library for tokenizing textual descriptions.
3. Instantiate the dataset class with the tokenizer object and desired parameters.
4. Define the image encoder, text encoder, projection head, and CLIP model using the provided components.
5. Train the model using appropriate training data.
6. After training, use the inference functions to retrieve relevant images based on textual queries.

## Contributors

- Ahsan Bilal 
- Muhammad Haseeb Nizami

## License

This project is licensed under the [MIT License](LICENSE).
