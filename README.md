# Disaster Response Pipeline Project

## Project Overview
A Machine Learning Pipeline is created which classifies the real messages that were sent during disaster events. The **intent** of this project is to help the affected people by sending the **categorized** messages to the appropriate disaster relief agency. This Project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This disaster data has been collected from [Figure Eight](https://www.figure-eight.com/).

## Project Components
1. ETL Pipeline
The Python script, `process_data.py`, contains a **data cleaning** pipeline which:

    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database

2. ML Pipeline
The Python script, `train_classifier.py`, contains a **machine learning** pipeline which:

    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file

3. Flask Web App
- Once you run the Web App as per the instructions written below, you will see a page with a space to add your text message and two visualizations. 
    1. Distribution of Message Genres (Count of each Genre)
    2. Distribution of Message Category-Names (Count of each Category-name)

- Also, you will see a text holder to write your message, once you input your message, the model will run and give you the categorization of the your message, firstly by highlighting if the message is related or not.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
