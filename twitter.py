from sklearn.svm import LinearSVC
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from util import lemmatizer, vectorizer, dictionary
from util.lemmatizer import pos_dict


if __name__ == '__main__':

    df = pd.read_csv('Train.csv', header=0)

    encoder = LabelEncoder()
    label_sentiment = encoder.fit_transform(df['Sentiment'])
    label_topic = encoder.fit_transform(df['Topic'])
    label = label_topic
    train = []
    clean = dictionary.clean()
    lemmatizer = lemmatizer.create_lemmatizer(train, clean, df)
    ngrams_vectorizer = vectorizer.create_vectorizer()
    feature = ngrams_vectorizer.fit_transform(train)

    clf = LinearSVC()
    clf.fit(feature, label)

    df_test = pd.read_csv('TestingSamples/Test.csv', header=0)
    print("You can add files to testing set in folder TestingSamples. Enter filenames one by one, ot enter 'none' if "
          "you want to stop")
    while True:
        filename = input()
        if filename == "none":
            break
        df_test.append(pd.read_csv('TestingSamples' + filename, header=0))
    test = []
    for doc in df_test['TweetText']:
        words = []
        for (word, pos) in pos_tag(word_tokenize(clean.fit(doc))):
            if word not in stopwords.words('english'):
                words.append(lemmatizer.lemmatize(word, pos_dict.get(pos[0].upper(), wordnet.NOUN)))
        test.append(' '.join(words))

    label_sentiment = encoder.fit_transform(df_test['Sentiment'])
    label_topic = encoder.fit_transform(df_test['Topic'])
    label_test = label_topic
    feature_test = ngrams_vectorizer.transform(test)
    pred_test = clf.predict(feature_test)
    print(classification_report(label_test, pred_test))

