Emotion Detection from Text using Deep Learning (LSTM + GloVe)
📂 Project Structure

emotion-detection-text/
├── README.md
├── data/
│   └── emotions.csv
├── notebooks/
│   └── emotion_detection_lstm.ipynb
├── models/
│   └── lstm_emotion_model.h5
├── requirements.txt
└── app.py

🧠 Project Summary

A deep learning model that detects emotions (joy, anger, sadness, etc.) in text data using an LSTM-based model and GloVe word embeddings. Trained on the Emotion Dataset, and deployable as a Flask web API.
✅ Features

    Multiclass classification (6 emotions)

    LSTM + pre-trained GloVe embeddings

    Clean training pipeline with visualization

    Flask API for inference

    Ready for deployment (Heroku or local)

    Modular structure for easy improvements

📦 Requirements (requirements.txt)

numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow
keras
flask
nltk


📓 Jupyter Notebook (notebooks/emotion_detection_lstm.ipynb)

Steps inside the notebook:

    Load and preprocess the dataset (tokenization, label encoding)

    Load GloVe embeddings

    Build LSTM model with embedding layer

    Train and evaluate (accuracy, confusion matrix)

    Save model to models/lstm_emotion_model.h5
Example 
Input:

{ "text": "I am feeling awesome today!" }

Output:

{ "emotion": "joy" }
Model

    LSTM architecture with dropout

    GloVe (100d) embedding layer

    Trained on 6-class dataset

📁 Folder Structure

    data/ - input CSV dataset

    models/ - saved model and tokenizer

    notebooks/ - Jupyter training code

    app.py - Flask API
  #Future Improvements

    Add attention mechanism

    Support longer texts (news, tweets)

    Train on multilingual data
