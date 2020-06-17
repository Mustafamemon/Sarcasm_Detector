import os, re,sys

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics, svm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sentistrength import PySentiStr

from FeatureCount import SequencePunctuationCount ,\
                         SequenceCharacterCount , \
                         CapitalizedCount , \
                         ExclamationCount , \
                         EmojiCount 
from senticnet.senticnet import SenticNet

DIR_PATH = os.path.dirname(os.path.abspath(""))
PICKLES_PATH = os.path.join(DIR_PATH, "Pickles")
CODE_PATH = os.path.join(DIR_PATH, "Code")


# nltk.download('punkt')
# nltk.download('wordnet')


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(debug=False)

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def welcome():
    return {"message": "Welcome, Sarcasm Detector!"}


@app.get("/is_sarcastic/{sentence}")
def is_sarcastic(sentence: str):
    _is_sarcastic = False
    _is_sarcastic = bool(pre_process_and_predict(sentence))
    return {"input_sentence": sentence, "is_sarcastic": _is_sarcastic}

@app.get("/is_sarcastic/")
def is_sarcastic_query(sentence: str):
    _is_sarcastic = False
    _is_sarcastic = bool(pre_process_and_predict(sentence))
    return {"input_sentence": sentence, "is_sarcastic": _is_sarcastic}


def pre_process_and_predict(sentence):
    wordnet_lemmatizer = WordNetLemmatizer()
    # # Replacing double quotes with single, within a string
    sentence = sentence.replace("\"", "\'")
    # # Removing unnecessary special characters, keeping only ,  ! ? 
    sentence = re.sub(r"[^!?,a-zA-Z0-9\ ]+", '', sentence)
    # # Lemmatization on verbs
    sentence = ' '.join([wordnet_lemmatizer.lemmatize(word, pos='v') for word in word_tokenize(sentence)])

    sn = SenticNet()
    senti = PySentiStr()
    senti.setSentiStrengthPath(CODE_PATH + '/sentistrength/SentiStrength.jar')
    senti.setSentiStrengthLanguageFolderPath(CODE_PATH +'/sentistrength/SentStrength_Data/')

    sentiment_score = []

    for sen in sent_tokenize(sentence):
        senti_pos , senti_neg = senti.getSentiment(sen,score='dual')[0]
        senti_pos -= 1
        if senti_neg == -1: 
            senti_neg = 0
        sum_pos_score = 0
        sum_neg_score = 0
        for word in word_tokenize(sen):
            try:
                w_score = float(sn.polarity_intense(word))*5
            except KeyError:
                w_score = 0
            if w_score > 0:
                sum_pos_score = sum_pos_score + w_score
            elif w_score < 0:
                sum_neg_score = sum_neg_score + w_score
        sum_pos_score = (sum_pos_score+senti_pos)/2
        sum_neg_score = (sum_neg_score+senti_neg)/2
        sentiment_score.append((sum_pos_score , sum_neg_score))
    additional_features_s  = [] 
    additional_features_ns = [] 

    contra     = []
    pos_low    = []
    pos_medium = []
    pos_high   = []
    neg_low    = []
    neg_medium = []
    neg_high   = []

    for sum_pos_score,sum_neg_score in sentiment_score:
        contra.append(int(sum_pos_score > 0 and abs(sum_neg_score) > 0))
        pos_low.append(int(sum_pos_score < 0))
        pos_medium.append(int(sum_pos_score >= 0 and sum_pos_score <= 1))
        pos_high.append(int(sum_pos_score >= 2))
        neg_low.append(int(sum_neg_score < 0))
        neg_medium.append(int(sum_neg_score >= 0 and sum_neg_score <= 1))
        neg_high.append(int(sum_neg_score >= 2))
    additional_features_s = additional_features_s + [max(pos_medium),max(pos_high),max(neg_medium),max(neg_high)]
    additional_features_ns = additional_features_ns + [max(pos_low),max(neg_low)]

    tweet             = sentence
    punctuation_count = SequencePunctuationCount(tweet)
    character_count   = SequenceCharacterCount(tweet)
    capitalized_count = CapitalizedCount(tweet)
    exclamation_count = ExclamationCount(tweet)
    #     emoji_count       = EmojiCount(tweet)
    f_count           = [punctuation_count,character_count,capitalized_count,exclamation_count] 
    for count in f_count:
        f_low    = int(count == 0 )
        f_medium = int(count >= 1 and count <= 3 )
        f_high   = int(count >= 4)
        additional_features_s = additional_features_s + [f_medium,f_high]
        additional_features_ns = additional_features_ns + [f_low]
    X = [sentence]

    in_file = open(os.path.join(PICKLES_PATH, "vocab.pickle"), "rb")
    vocab = pickle.load(in_file)
    in_file.close()

    in_file = open(os.path.join(PICKLES_PATH, "model.pickle"), "rb")
    model = pickle.load(in_file)
    in_file.close()


    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(X)
    ans = int(sum(model.predict(X)))
    print('Sentence : ',sentence)
    print('Sarcastic features : ',additional_features_s)
    print('Not Sarcastic features : ',additional_features_ns)
    print('Contradict : ',max(contra))
    print('Model Predict : ',ans)
    print('My obs : ',int((sum(additional_features_s)  >= sum(additional_features_ns)) and max(contra) == 1))
    print('Final Prd : ',end='')

    if ans == 1 or ((sum(additional_features_s)  >= sum(additional_features_ns)) and max(contra) == 1):
        return True
    else:
        return False


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
    