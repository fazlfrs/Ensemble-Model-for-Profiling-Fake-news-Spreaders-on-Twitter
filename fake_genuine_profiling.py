import argparse
import os
import codecs
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import preprocessor as p
from xml.dom import minidom
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC 
from sklearn import preprocessing
import xml.etree.cElementTree as ET
from collections import Counter 
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from sklearn import utils
from sklearn.linear_model import LogisticRegression
import multiprocessing

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to dataset directory")
    parser.add_argument('-o', '--output', help="path to output directory")
    parser.add_argument('-t', '--train_dir', help="path to train dataset directory")
    parser.add_argument('-m', '--models', help="path to models directory")
    parser.add_argument('-n', type=int, help="n-gram order (default=4)", default=4)
    args = parser.parse_args()
    if args.input is None and args.output is None and args.train_dir is None and args.models:
        parser.print_usage()
        exit()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def preprocessing1(tweets, _stopwords):
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

def preprocessing2(tweets, _stopwords):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG
                  , p.OPT.SMILEY, p.OPT.NUMBER, p.OPT.RESERVED)
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = p.clean(tweets)
    word_list = cleaned_tweets.split()    
    mod_tweet = []
    for word in word_list:
        if word.lower() in _stopwords:
            continue
        mod_tweet.append(lemmatizer.lemmatize(word.lower()))      
    return ' '.join(mod_tweet)

def get_tweets(xml_path):
    content = open(xml_path).read()
    tweets = []
    i = 0
    while True:
        i += 1
        start_documents = content.find('<document>')
        end_documents = content.find('</document>')
        tweets.append(' '.join(content[start_documents + 19 : end_documents-3].split()))
        content = content[end_documents+10:]
        if i == 100:
            break
    return ' '.join(tweets)

def loading_datasets(fake, genuine):
    with codecs.open(fake, 'r', encoding='utf-8') as f:
        fake_tweets = f.read().split('\n')
    with codecs.open(genuine, 'r', encoding='utf-8') as f:
        genuine_tweets = f.read().split('\n')
    return fake_tweets, genuine_tweets

def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors

def save_xml(path, user, l, t):
    author = ET.Element("author" , id=str(user)[:-4], lang=l, type=t)
    tree = ET.ElementTree(author)
    tree.write(path)

def profiling():
    stopwords_list = {'en': set(stopwords.words('english')) , 'es': set(stopwords.words('spanish')) }
    max_abs_scaler = preprocessing.MaxAbsScaler()
    args = get_args()
    n = args.n
    input_dir = os.path.normpath(args.input)
    out_dir = os.path.normpath(args.output)
    mkdir(out_dir)
    for language_dir in os.listdir(input_dir):
        
        print("Working on: ", language_dir)
        users_input_dir = os.path.join(input_dir, language_dir)
        users_out_dir = os.path.join(out_dir, language_dir)
        train_set_dir = os.path.join(args.train_dir, language_dir)
        models_dir = os.path.join(args.models, language_dir)
        mkdir(users_out_dir)

        #Fake and Genuine Estimator and Vectorizer
        fake, genuine = loading_datasets( os.path.join(train_set_dir,'train_fake.txt'), 
                                    os.path.join(train_set_dir,'train_genuine.txt')) 
        
        # Training Fake and Genuine Estimators for Fake and Genuine Prediction ....
        
        print("\t Training Fake and Genuine Estimators for Fake and Genuine Prediction .... ")
        fake_genuine_train_set = fake + genuine
        fake_genuine_train_labels = ['fake' for _ in fake] + ['genuine' for _ in genuine]

        #fake_genuine TFIDF
        print("\t Training Estimator1 ...")
        tfidf_fake_genuine_train_set = [preprocessing2(text, stopwords_list[language_dir]) for text in fake_genuine_train_set]
        tfidf_fake_genuine_vocab = pickle.load(open( os.path.join( models_dir , "tfidf_fake_genuine.pkl"),"rb"))
        tfidf_fake_genuine_vectorizer  = TfidfVectorizer(decode_error="replace", vocabulary= tfidf_fake_genuine_vocab, norm='l2', \
                                                         strip_accents=False, sublinear_tf=True )
        tfidf_fake_genuine_train_data = tfidf_fake_genuine_vectorizer.fit_transform(tfidf_fake_genuine_train_set)
        tfidf_scaled_fake_genuine_train_data = max_abs_scaler.fit_transform(tfidf_fake_genuine_train_data)
        fake_genuine_estimator1 = LinearSVC(C=0.01)
        fake_genuine_estimator1.fit(tfidf_scaled_fake_genuine_train_data, fake_genuine_train_labels)
                
        print("\t  Training Estimator1 -----------------done \n")
        
        
        #fake_genuine DOC2VEC
        print("\t Training Estimator2 - doc2vec...")
        cores = multiprocessing.cpu_count()
        doc2vec_fake_genuine_train_set = [TaggedDocument(words=tfidf_fake_genuine_train_set[i].split(), tags=[fake_genuine_train_labels[i]]) for i in range(0, len(tfidf_fake_genuine_train_set))]
        doc2vec_model = Doc2Vec.load(os.path.join(models_dir, "doc2vec_fake_genuine.d2v"))
        y_train, x_train = vector_for_learning(doc2vec_model, doc2vec_fake_genuine_train_set)
        doc2vec_scaled_train_data = max_abs_scaler.fit_transform(x_train)
        print("\n\n ==========================\n")
        fake_genuine_estimator2 = LogisticRegression(n_jobs=1, C=1e5)     ##C=0.01 , dual=True)
        fake_genuine_estimator2.fit(doc2vec_scaled_train_data, y_train)
        print("\t  Training Estimator2 -----------------done \n")
                
        #fake_genuine NGRAM
        print("\t Training Estimator 3...")
        ngram_fake_genuine_vocab = pickle.load(open( os.path.join( models_dir , "ngram_fake_genuine.pkl"),"rb"))
        ngram_fake_genuine_vectorizer = CountVectorizer(strip_accents=False, analyzer='char', ngram_range=(n,n),
                                            lowercase=False,vocabulary=ngram_fake_genuine_vocab)
        ngram_fake_genuine_train_data = ngram_fake_genuine_vectorizer.fit_transform(tfidf_fake_genuine_train_set)
        ngram_fake_genuine_train_data = ngram_fake_genuine_train_data.astype(float)
        for i in range(len(tfidf_fake_genuine_train_set)):
            ngram_fake_genuine_train_data[i]=ngram_fake_genuine_train_data[i]/len(tfidf_fake_genuine_train_set[i])
        ngram_scaled_train_data = max_abs_scaler.fit_transform(ngram_fake_genuine_train_data)
        fake_genuine_estimator3 = LinearSVC(C=0.01)
        fake_genuine_estimator3.fit(ngram_scaled_train_data, fake_genuine_train_labels)
        
        print("\t   Training Estimator 3 --------- done \n")
        
         
        for user in os.listdir(users_input_dir):
            print("\t\t Working on user: ", user )
            user_tweets = get_tweets(os.path.join(users_input_dir, user))
            cleaned_tweet = preprocessing1(user_tweets , stopwords_list[language_dir])
            
            f1 = tfidf_fake_genuine_vectorizer.fit_transform([cleaned_tweet])
            scaled_f1 = max_abs_scaler.fit_transform(f1)
            p1 = fake_genuine_estimator1.predict(scaled_f1)[0]
            
            cleaned_tweet = preprocessing2(user_tweets, stopwords_list[language_dir])
            data = [TaggedDocument(words=cleaned_tweet.split(),tags=[' '])]
            t2,f2 = vector_for_learning(doc2vec_model, data)
            data = vector_for_learning(doc2vec_model, data )
            scaled_f2 = max_abs_scaler.fit_transform(f2)
            p2 = fake_genuine_estimator2.predict(scaled_f2)[0]
            
            f3 = ngram_fake_genuine_vectorizer.fit_transform([cleaned_tweet])
            f3 = f3.astype(float)
            f3[0] = f3[0]/len(cleaned_tweet)
            scaled_f3 = max_abs_scaler.fit_transform(f3)
            p3 = fake_genuine_estimator3.predict(scaled_f3)[0]
            
            #occurence_count = Counter([p1 , p2 , p3]) 
            occurence_count = Counter([p1, p2, p3]) 
            pred = occurence_count.most_common(1)[0][0]
            save_xml(os.path.join(users_out_dir, user), user, str(language_dir), str(pred))
            print("\t\t Results saved to " , str(os.path.join(users_out_dir, user)))
            print("\t\t -----------------------------------------------")
        
profiling()   
