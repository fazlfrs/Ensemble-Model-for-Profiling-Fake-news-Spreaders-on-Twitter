import argparse
import os  
import sys
import codecs
import preprocessor as p
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to prepared dataset")
    parser.add_argument('-o', '--output', help="path to output directory", default='pre-trained_models')
    parser.add_argument('-ft', type=int, help="frequency threshold (default=5)", default=5)
    args = parser.parse_args()
    if args.input is None:
        parser.print_usage()
        sys.exit()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def loading_datasets(fake, genuine):
    with codecs.open(fake, 'r' ,encoding='utf-8') as f:
        fake_tweets = f.read().split('\n')
    with codecs.open(genuine, 'r' ,encoding='utf-8' ) as f:
        genuine_tweets = f.read().split('\n')
    return fake_tweets, genuine_tweets

def Preprocessing(tweets, _stopwords):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.RESERVED)
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = p.clean(tweets)
    word_list = cleaned_tweets.split()    
    mod_tweet = []
    for word in word_list:
        if word.lower() in _stopwords:
            continue
        mod_tweet.append(lemmatizer.lemmatize(word.lower()))      
    return ' '.join(mod_tweet)

def represent_tweet(tweets):
    tokens = TweetTokenizer().tokenize(tweets)
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def extract_vocabulary(tweets , freqthrs):
    occurrences=defaultdict(int)
    for tweet in tweets:
        tweet_occurrences = represent_tweet(tweet)
        for ngram in tweet_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=tweet_occurrences[ngram]
            else:
                occurrences[ngram]=tweet_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=freqthrs:
            vocabulary.append(i)
    return vocabulary

def training():
    stopwords_list = {'en': set(stopwords.words('english')) , 'es': set(stopwords.words('spanish')) }
    args = get_args()
    freqthrs = args.ft
    input_dir = os.path.normpath(args.input)
    out = os.path.normpath(args.output)
    mkdir(out)
    for dir in os.listdir(input_dir):
        print("Working on Language: ", dir)
        out_dir = os.path.join(out, dir)
        mkdir(out_dir)
        fake, genuine = loading_datasets(os.path.join(input_dir, dir,'train_fake.txt'), 
                                    os.path.join(input_dir,dir,'train_genuine.txt'))
        
        print("\t Fake train-set size: ", len(fake))
        print("\t Genuine train-set size: ", len(genuine))
        
        #training for fake and genuine
        print("\t Preparing Fake & Genuine dataset for tfidf vectorizer .... ")
        train_set = fake + genuine
        train_labels = ['fake' for _ in fake] + ['genuine' for _ in genuine]
        tfidf_train = [Preprocessing(text, stopwords_list[dir]) for text in train_set]
     
        print("\t Extracting Fake & Genuine vocabulary .... ")
        vocab = extract_vocabulary(tfidf_train, freqthrs )
        
        print("\t Train TF-IDF for Fake & Genuine Profiling .... ")
        vectorizer  = TfidfVectorizer(vocabulary=vocab, norm='l2', strip_accents=False, sublinear_tf=True)
        tfidf_train_data = vectorizer.fit_transform(tfidf_train)
        pickle.dump(vectorizer.vocabulary_, open(os.path.join(out_dir, "tfidf_fake_genuine.pkl"),"wb"))
        print("\tTrained TF-IDF vocabulary saved to :", str(os.path.join(out_dir, "tfidf_fake_genuine.pkl")))

        print('\t ------------------------------------')

training()