import numpy
import pandas
import re
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from gensim.models import Doc2Vec
from tqdm import tqdm
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

def removeAndCut(line) :
    line = str(line)
    if line.strip() == '' :
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\s]")
    line = rule.sub('', line)
    return line.split(' ')

def preprocess(data) :
    data = data[['review', 'sentiment']]
    data.loc[:, 'review'] = data['review'].apply(removeAndCut)
    print(data.head())
    print("Preprocess End")
    return data

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    print("Vec_for_learning End")
    return targets, regressors

if __name__ == '__main__' :
    data = pandas.read_csv('data/train.csv')
    data = preprocess(data)

    test = pandas.read_csv('data/test_data.csv')
    test = preprocess(test)

    # #创建训练集和测试集
    # train, test = train_test_split(data, test_size=0.3, random_state=42,stratify = data.sentiment.values)
    
    #创建标签化文档
    train_tagged = data.apply(
        lambda r: TaggedDocument(words=r['review'], tags=[r['sentiment']]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=r['review'], tags=[r['sentiment']]), axis=1)

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0,  negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
    print("Train dbow End")

    y_train, X_train = vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = vec_for_learning(model_dbow, test_tagged)

    logreg = LogisticRegression(n_jobs=1, C=1e5)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    # print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    # print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

    test = pandas.read_csv('data/submission.csv')
    test.loc[:, 'sentiment'] = y_pred
    test.to_csv('result.csv', index = False)