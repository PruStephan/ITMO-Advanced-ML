from sklearn.feature_extraction import text
from sklearn.pipeline import FeatureUnion, Pipeline

def create_vectorizer():
    word_vectorizer = text.TfidfVectorizer(
        analyzer='word', ngram_range=(1, 3),
        min_df=2, use_idf=True, sublinear_tf=True)
    char_vectorizer = text.TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5),
        min_df=2, use_idf=True, sublinear_tf=True)
    ngrams_vectorizer = Pipeline(
        [('feats', FeatureUnion([('word_ngram', word_vectorizer), ('char_ngram', char_vectorizer)]))])

    return ngrams_vectorizer
