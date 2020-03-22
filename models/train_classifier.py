# import the necessary libraries
import sys
import pandas as pd
import numpy as np
import re

import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    INPUT: 
    database_filepath (str): path to the database file
    
    OUTPUT:
    X (series, str): contains the messages
    y (dataframe): consists of all the classes
    category_names: names of the classes (target variables)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('msgCat', con=engine)
    df.dropna(inplace=True)
    X = df['message']
    y = df.iloc[:, 4:]
    return X, y, y.columns


def tokenize(text):
    """
    INPUT: 
    text (str): message strings which need to be cleaned
    
    OUTPUT:
    clean_tokens: tokens of the messages which are cleaned
    """
    # remove punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    # lemmatize as shown in the classroom
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    # remove extra spaces if any and convert all the strings to lowercase
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    To Create (initialize) a Model
    """
    model_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    # not using GridSearchCV in this part because this pipeline has been chosen from the jupyter notebook
    # this pipeline gave better results compared to the rest models
    return model_pipeline
 
    
def evaluate_model(model, X_test, y_test, category_names):
    """
    To return F1-score, precision, and recall for test_set of all categories
    and overall scores also
    """
    y_pred = model.predict(X_test)
    results = pd.DataFrame(columns=['Category', 'f1_score', 'precision', 'recall'])
    num = 0
    for category in y_test.columns:
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[category], 
                                                                              y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', category)
        results.set_value(num+1, 'f1_score', f1_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Overall f1_score:', results['f1_score'].mean())
    print('Overall precision:', results['precision'].mean())
    print('Overall recall:', results['recall'].mean())
    print(results)


def save_model(model, model_filepath):
    """
    To save a pickle file of the trained model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()