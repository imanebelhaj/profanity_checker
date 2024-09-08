# Insult Detection Chatbot Model

This project focuses on developing a Natural Language Processing (NLP) model to detect insults in text data. The model has been integrated into a chatbot system to identify and classify insulting language across multiple languages. It is primarily trained to handle French and English datasets with high accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- -[Project Strcture](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)


## Project Overview

The insult detection chatbot leverages machine learning algorithms and NLP techniques to classify text into insulting or non-insulting categories. It aims to improve chatbot user interactions by identifying inappropriate language and responding accordingly. This project includes both the model development and integration of the insult detection model into a RESTful API using Flask.

## Features

- Insult detection for both English and French text.
- Integrated into a chatbot interface.
- Achieved **99.7% accuracy** for French and **94.2% accuracy** for English in insult classification.
- Provides precision, recall, and F1-score for model evaluation.
- Deployed using a REST API for easy integration with front-end applications.

## Technologies Used

- **Python**: For data processing, model building, and deployment.
- **Flask**: For API development and integration.
- **Scikit-Learn**: For building machine learning models.
- **Pandas**: For data manipulation and cleaning.
- **Natural Language Toolkit (NLTK)**: For NLP tasks like tokenization and text processing.
- **Seaborn & Matplotlib**: For data visualization.
- **VS Code**: Development environment.

## Model Performance

The model was trained and evaluated on a custom dataset of insults and non-insults, resulting in the following performance metrics:


### French Dataset:
- **Accuracy**: 99.7%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%

### English Dataset:
- **Accuracy**: 94.2%
- **Precision**: 94%
- **Recall**: 94%
- **F1-Score**: 94%



## Project Strcuture:
Project Structure:
.
├── profanity_checker_app/            # Contains the main application files
│   ├── app.py                        # Main Flask app file
│   ├── model/                        # contains the last model :best_model.pkl ; tfidf_vectorizer.pkl
│   ├── static/                       # Typically used for CSS, JS, and other static resources
│   └── templates/                    # HTML templates for the web interface
│
├── profanity_check_best_model/        # Includes model files and scripts related to the profanity detection model ( all the files used when working on data and ML )
│   ├── best_model.pkl                # The trained model
│   ├── fr_en_model.py                # Python script for loading the French-English model
│   ├── test_english.py               # Script for testing the English model
│   ├── test_fr_profanity.py          # Script for testing the French profanity model
│   └── tfidf_vectorizer.pkl          # The vectorizer used for transforming the text data
│
├── other/                            # Includes resources and files for different languages and chatbot functionalities
│   ├── arabic/                       # Contains Arabic language resources
│   ├── chatbot_front/                # Front-end resources for chatbot
│   ├── english/                      # Contains English language resources
│   └── french/                       # Contains French language resources



## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/insult-detection-chatbot.git
    cd insult-detection-chatbot
    ```

2. **Install the dependencies**:
    Flask
    pandas
    scikit-learn
    nltk
    numpy
    matplotlib
    seaborn
    Python



5. **Download the dataset**:
    The datasets used in this project are from publicly available sources. You can download datasets for insult detection or hate speech from platforms like [Kaggle Datasets](https://www.kaggle.com/datasets) or the [GitHub repository](https://github.com/aymeam/Datasets-for-Hate-Speech-Detection).

## Usage

Once everything is set up, you can run the Flask API and test the model 
## Future Work

- **Language Expansion**: Extend insult detection to other languages.
- **Model Optimization**: Experiment with deep learning models like RNN or Transformer-based models.
- **Improved User Interface**: Create a more interactive and user-friendly chatbot interface.
- **Real-time Deployment**: Host the model on cloud services for live insult detection.


