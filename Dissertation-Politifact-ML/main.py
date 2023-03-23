import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plot
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.svm import SVR
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import string


def combine_csv_files():
    # Combining all Politifact Data CSV files
    cols = ['Source', 'Headline', 'Source-Bio', 'Judgement-Text', 'Judgement-Author', 'Judgement-Text-URL', 'Rating']

    df1 = pd.read_csv("Updated_Politifacts_0-19.csv", encoding='unicode_escape')
    df1.columns = cols

    df2 = pd.read_csv("Updated_Politifacts_20-194.csv", encoding='unicode_escape')
    df2.columns = cols

    df3 = pd.read_csv("Updated_Politifacts_214-220.csv", encoding='unicode_escape')
    df3.columns = cols

    df4 = pd.read_csv("Updated_Politifacts_221-230.csv", encoding='unicode_escape')
    df4.columns = cols

    df5 = pd.read_csv("Updated_Politifacts_231-282.csv", encoding='unicode_escape')
    df5.columns = cols

    df6 = pd.read_csv("Updated_Politifacts_283-359.csv", encoding='unicode_escape')
    df6.columns = cols

    df7 = pd.read_csv("Updated_Politifacts_360-625.csv", encoding='unicode_escape')
    df7.columns = cols

    frames = [df1, df2, df3, df4, df5, df6, df7]
    df = pd.concat(frames)

    # Creating a new column called "Headline-And-Source-Bio-Text" which contains the 'Headline' text and the 'Source-Bio' text
    df["Headline-And-Source-Bio-Text"] = df["Headline"] + " " + df['Source-Bio']

    # Simple text pre-processing that removes punctutation and lower cases the text
    df['Headline-And-Source-Bio-Text'] = df['Headline-And-Source-Bio-Text'].str.replace('[{}]'.format(string.punctuation), '')
    df['Headline-And-Source-Bio-Text'] = df['Headline-And-Source-Bio-Text'].str.lower()
    df['Judgement-Text'] = df['Judgement-Text'].str.replace('[{}]'.format(string.punctuation), '')
    df['Judgement-Text'] = df['Judgement-Text'].str.lower()


    # Remove Flip-O-Meter Data
    df = df[~df.Rating.isin(['full-flop', 'half-flip', 'no-flip'])]

    # Map ratings to numeric value
    raw_ratings = df.iloc[:, 6]
    ratings = mapping_ratings(raw_ratings)
    df.Rating = ratings

    # Remove Null Values
    clean_df = df.dropna()

    # Print Number of Items in Data
    print(f'Number of items in the dataframe: {len(clean_df)}')

    # Output cleaned data to new csv file
    clean_df.to_csv("Updated_Politifacts_Data.csv", index=False)


def text_pre_processing():
    # Read in Politifacts Data
    facts_text = pd.read_csv("Updated_Politifacts_Data.csv")

    # StopWords and Stemmer
    global englishStopWords, stemmer
    englishStopWords = stopwords.words('english')
    stemmer = PorterStemmer()

    # Tokenizer
    def tok(text):
        tokens = nltk.word_tokenize(text)
        stems = [stemmer.stem(token) for token in tokens]
        return [w for w in stems if (w.isalpha() and w.isascii() and len(w) > 2) and w not in englishStopWords]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tok, max_features=1000)
    X = vectorizer.fit_transform(facts_text['Headline-And-Source-Bio-Text'])
    feature_names = vectorizer.get_feature_names_out()

    # Output Word Features to dataframe and csv
    df = pd.DataFrame(X.toarray(), columns=[feature_names])
    df.to_csv("Updated_Politifacts_Word_Features_1000.csv", index=False)


def mapping_ratings(df_column):
    array = df_column.to_numpy()
    result_array = []
    for i in range(len(array)):
        if array[i] == 'pants-fire':
            result_array.append(0)
            # result_array.append(0)
        elif array[i] == 'FALSE' or array[i] == 'false':
            result_array.append(1)
            # result_array.append(0)
        elif array[i] == 'barely-true':
            result_array.append(2)
            # result_array.append(0)
        elif array[i] == 'half-true':
            result_array.append(3)
            # result_array.append(1)
        elif array[i] == 'mostly-true':
            result_array.append(4)
            # result_array.append(1)
        elif array[i] == 'TRUE' or array[i] == 'true':
            result_array.append(5)
            # result_array.append(1)
    return result_array


def sentiment_analysis():
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    clean_data = pd.read_csv("Updated_Politifacts_Data.csv")
    judgement_text = clean_data.iloc[:, 3]
    sentiment_values = []
    sentiment_labels = []
    total_positive_words = []
    total_negative_words = []

    for i in range(len(judgement_text)):
        text = judgement_text.iloc[i]
        processed_text = nlp(text)
        sentiment = processed_text._.blob.polarity
        sentiment = round(sentiment, 2)
        if sentiment > 0:
            sent_label = 1
        else:
            sent_label = 0
        sentiment_values.append(sentiment)
        sentiment_labels.append(sent_label)

        positive_words = []
        negative_words = []
        for x in processed_text._.blob.sentiment_assessments.assessments:
            if x[1] > 0:
                positive_words.append(x[0][0])
            elif x[1] < 0:
                negative_words.append(x[0][0])
            else:
                pass
        total_positive_words.append(', '.join(set(positive_words)))
        total_negative_words.append(', '.join(set(negative_words)))

    clean_data['Sentiment Values'] = sentiment_values
    clean_data['Sentiment Labels'] = sentiment_labels
    clean_data['Positive Words'] = total_positive_words
    clean_data['Negative Words'] = total_negative_words

    clean_data.to_csv("Updated_Politifacts_With_Sentiment_Feats.csv", index=False)


def machine_learning_processes():
    print('Pre-Processing Data...')

    # Word Features
    # clean_data = pd.read_csv("Politifacts.csv")
    # word_features = pd.read_csv("Politifacts_Word_Features.csv")
    # word_features = pd.read_csv("new_politifacts.csv")
    # X = np.column_stack((word_features.iloc[:, 0], word_features.iloc[:, 1], word_features.iloc[:, 2],
    #                      word_features.iloc[:, 3], word_features.iloc[:, 4], word_features.iloc[:, 5],
    #                      word_features.iloc[:, 6], word_features.iloc[:, 7], word_features.iloc[:, 8],
    #                      word_features.iloc[:, 9]))
    # y = clean_data.iloc[:, 3]
    # kNN(X, y, 'word_features')
    # lasso(X, y, 'word_features')

    # Sentiment Analysis
    # clean_data = pd.read_csv("Politifacts_With_Sentiment_Feats.csv")
    # X = np.column_stack((clean_data.iloc[:, 5], clean_data.iloc[:, 6]))
    # y = clean_data.iloc[:, 3]
    # kNN(X, y, 'sentiment_analysis')
    # lasso(X, y, 'sentiment_analysis')

    # SVR - Word Features and Sources in One-Hot-Encoding format
    # clean_data = pd.read_csv("Updated_Politifacts_Data.csv")
    # word_features = pd.read_csv("Updated_Politifacts_Word_Features_1000.csv")
    # sources = clean_data.iloc[:, 0]
    #
    # col_arrays = []
    # for col in word_features.columns:
    #     col_arrays.append(word_features[col].values)
    #
    # # One-Hot Encoding
    # encoded_sources = pd.get_dummies(sources)
    # for source in encoded_sources.columns:
    #     col_arrays.append(encoded_sources[source].values)
    #
    # ## Label Encoding
    # # unique_sources = sources.unique()
    # # mapped_sources = map_sources(sources, unique_sources)
    # # col_arrays.append(mapped_sources)
    #
    # X = np.column_stack(col_arrays)
    # y = clean_data.iloc[:, 6]
    # svr(X, y)

    # SVR - Sentiment Label Feature
    clean_data = pd.read_csv("Updated_Politifacts_With_Sentiment_Feats.csv")
    word_features = pd.read_csv("Updated_Politifacts_Word_Features_1000.csv")
    sources = clean_data.iloc[:, 0]

    col_arrays = []
    # for col in word_features.columns:
    #     col_arrays.append(word_features[col].values)
    #
    # # One-Hot Encoding
    # encoded_sources = pd.get_dummies(sources)
    # for source in encoded_sources.columns:
    #     col_arrays.append(encoded_sources[source].values)

    col_arrays.append(clean_data.iloc[:, 8])
    X = np.column_stack(col_arrays)
    y = clean_data.iloc[:, 6]
    svr(X, y)


# kNN Model, Cross-Validation and 2D Scatter Plot
def kNN(X, y, type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    kNN_cross_validation(X, y)
    knn_model(X, y, X_train, X_test, y_train, y_test, type)


# Lasso Model, Cross-Validation and 2D Scatter Plot
def lasso(X, y, type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    lasso_cross_validation(X, y)
    lasso_model(X, y, X_train, X_test, y_train, y_test, type)


# SVR Model, Cross-Validation and 2D Scatter Plot
def svr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    # svr_cross_validation(X, y)
    svr_model(X, y, X_train, X_test, y_train, y_test)


def kNN_cross_validation(X, y):
    # 5-Fold Cross Validation for kNN
    std_error = []
    mean_error = []
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 51, 101]
    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k, weights='uniform')
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], y_pred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plot.rc("font", size=18)
    plot.rcParams["figure.constrained_layout.use"] = True
    plot.errorbar(k_values, mean_error, yerr=std_error, linewidth=3, ecolor="y")
    plot.title('5 Fold Cross-Validation for kNN')
    plot.xlabel("k")
    plot.ylabel("MSE")
    plot.show()


def knn_model(X, y, X_train, X_test, y_train, y_test, type):
    if type == 'word_features':
        model = KNeighborsRegressor(n_neighbors=21, weights="uniform").fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('MSE for kNN', mse)
        print('R2 for kNN', r2)

        plot.scatter(X[:, 0], y, marker='o', c='b', label='Training')
        plot.scatter(X_test[:, 0], y_pred, marker='o', c='r', label='Prediction')
        plot.xlabel("Word-Feature = Facebook")
        plot.ylabel("Truth Rating")
        plot.title('kNN for Word Features: Predictions and Training Data')
        plot.legend(loc='lower right')
        plot.show()

    elif type == 'sentiment_analysis':
        model = KNeighborsRegressor(n_neighbors=21, weights="uniform").fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('MSE for kNN', mse)
        print('R2 for kNN', r2)

        plot.scatter(X[:, 0], y, marker='o', c='b', label='Training')
        plot.scatter(X_test[:, 0], y_pred, marker='o', c='r', label='Prediction')
        plot.xlabel("Sentiment Value")
        plot.ylabel("Truth Rating")
        plot.title('kNN for Sentiment Score: Predictions and Training Data')
        plot.legend(loc='lower right')
        plot.show()


def lasso_cross_validation(X, y):
    std_error = []
    mean_error = []
    C_values = [1, 5, 10, 50, 100]
    for C in C_values:
        L1_penalty = 1 / (2 * C)
        model = Lasso(alpha=L1_penalty)
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], y_pred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plot.errorbar(C_values, mean_error, yerr=std_error, ecolor='y')
    plot.title('5 Fold Cross-Validation for Lasso Model')
    plot.xlabel('C')
    plot.ylabel('MSE')
    plot.show()


def lasso_model(X, y, X_train, X_test, y_train, y_test, type):
    if type == 'word_features':
        c = 50
        L1_penalty = 1 / (2 * c)
        model = Lasso(alpha=L1_penalty).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('Lasso Regression for C =', c)
        print('Intercept: ', model.intercept_)
        print('Coefficients: ', model.coef_, '\n')
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('MSE for Lasso', mse)
        print('R2 for Lasso', r2)

        plot.scatter(X[:, 7], y, marker='o', c='b', label='Training')
        plot.scatter(X_test[:, 7], y_pred, marker='o', c='r', label='Prediction')
        plot.xlabel("Word-Feature = Trump")
        plot.ylabel("Truth Rating")
        plot.title('Lasso for Word Features: Predictions and Training Data')
        plot.legend(loc='lower right')
        plot.show()

    elif type == 'sentiment_analysis':
        c = 10
        L1_penalty = 1 / (2 * c)
        model = Lasso(alpha=L1_penalty).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('Lasso Regression for C =', c)
        print('Intercept: ', model.intercept_)
        print('Coefficients: ', model.coef_, '\n')
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('MSE for Lasso', mse)
        print('R2 for Lasso', r2)

        plot.scatter(X[:, 0], y, marker='o', c='b', label='Training')
        plot.scatter(X_test[:, 0], y_pred, marker='o', c='r', label='Prediction')
        plot.xlabel("Sentiment Value")
        plot.ylabel("Truth Rating")
        plot.title('Lasso for Sentiment Score: Predictions and Training Data')
        plot.legend(loc='lower right')
        plot.show()


def svr_cross_validation(X, y):
    std_error = []
    mean_error = []
    C_values = [1, 5, 10, 50, 100]
    for C in C_values:
        model = SVR(C=C)
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], y_pred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plot.errorbar(C_values, mean_error, yerr=std_error, ecolor='y')
    plot.title('5 Fold Cross-Validation for SVR Model')
    plot.xlabel('C')
    plot.ylabel('MSE')
    plot.show()


def svr_model(X, y, X_train, X_test, y_train, y_test):
    print('Fitting Model...')
    model = SVR(C=10).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    print('MSE for SVR', mse)
    print('R2 for SVR', r2)
    print('F1 for SVR', f1)

    plot.scatter(X_train[:, X_train.shape[1] - 1], y_train, marker='o', c='b', label='Training')
    plot.scatter(X_test[:, X_test.shape[1] - 1], y_pred, marker='o', c='r', label='Prediction')
    plot.xlabel("Sentiment Label")
    plot.ylabel("Truth Rating")
    plot.title('SVR for Sentiment: Predictions and Training Data')
    plot.legend(loc='lower right')
    plot.show()


# Map sources to a unique variable
def map_sources(df_column, unique_sources):
    index_dict = {val: idx for idx, val in enumerate(unique_sources)}
    result_array = [index_dict[val] for val in df_column]
    return result_array


if __name__ == '__main__':
    combine_csv_files()
    # text_pre_processing()
    sentiment_analysis()
    machine_learning_processes()
