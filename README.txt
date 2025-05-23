Accent Detection Project
========================

 Goal:
This project detects English accents (American, British, Indian) from audio files using a neural network built with TensorFlow/Keras.

 Project Structure:
----------------------
Accent_detection_project/
  scripts/
         app.py                 # Streamlit UI for prediction
         train.py               # Trains the neural network
         predict.py             # Predicts accent from a single file
         utils.py               # Feature extraction utilities
         run_streamlit.py       # Launches Streamlit with browser
  models/                       # Stores trained model and label encoder
  data/                         # Accent folders with training audio(removed because space)
  reports/                      # Training report output
  requirements.txt              # All required packages
  README.txt                    # This file
  ffmpeg/                       #allows your app to read .mp3 files properly through librosa
  Accent-Detection-App.pptx     #the projcet presentation
  run_app.bat                   #to run the project

 Quick Start Guide
===================================

1.(Optional but recommended) Activate the virtual environment:
   
   .\venv\Scripts\activate(in powershell)
   

2.Install everything you need:
  
    pip install -r requirements.txt
   

3.you will find the ffmpeg folder in the project folder

     -Add the bin folder (C:\ffmpeg\bin) to your System PATH:
 
     -Search "Environment Variables" in Windows Search (if it didn't appear search it in control panal) 

     -Edit system environment variables

     -Click Environment Variables

     -Under System variables, select Path and then Edit

     -Add new entry: C:\ffmpeg\bin

     -Click OK and restart your terminal/IDE

4.Start the app:
   
  click on (run_app.bat) to start
   

5.Your browser will open — just upload a .wav or .mp3 file and the app will guess the accent.

 You don’t need to train anything. The model is already trained.

Enjoy testing the accent detector!


