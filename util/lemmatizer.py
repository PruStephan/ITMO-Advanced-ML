from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

pos_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

def create_lemmatizer(train, clean, df):
    lemmatizer = WordNetLemmatizer()
    for doc in df['TweetText']:
        words = []
        for word, pos in pos_tag(word_tokenize(clean.fit(doc))):
            if word not in stopwords.words('english'):
                words.append(lemmatizer.lemmatize(word, pos_dict.get(pos[0].upper(), wordnet.NOUN)))
        train.append(' '.join(words))
    return lemmatizer
