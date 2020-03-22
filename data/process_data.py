import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    INPUT: 
    messages_filepath (str): path to the file disaster_messages.csv
    categories_filepath (str): path to the file disaster_categories.csv
    
    OUTPUT:
    df (pd.DataFrame): merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    INPUT: 
    df (pd.DataFrame): merged dataframe
    
    OUTPUT:
    df (pd.DataFrame): cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(pat=';', expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to split the string at hyphen and
    # append the part before the hyphen to a list of column names
    category_colnames = []
    for i in range(len(row)):
        category_colnames.append(row[i].split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').apply(lambda x: x[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates if any
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    INPUT: 
    df (pd.DataFrame): cleaned dataframe
    database_filename (str): path to the database file
    
    OUTPUT:
    saves the dataframe to SQL database
    """
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('msgCat', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()