import numpy
import pandas
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def preprocess(data, mode) :
    data.loc[data['sex'] == 'male', 'sex'] = 1
    data.loc[data['sex'] == 'female', 'sex'] = 0
    data.loc[data['smoker'] == 'yes', 'smoker'] = 1
    data.loc[data['smoker'] == 'no', 'smoker'] = 0
    data.loc[:, 'ne'] = data['region'] == 'northeast'
    data.loc[:, 'nw'] = data['region'] == 'northwest'
    data.loc[:, 'se'] = data['region'] == 'southeast'
    data.loc[:, 'sw'] = data['region'] == 'southwest'
    del data['region']
    if (mode == 1) :
        data.loc[:, 'x0'] = numpy.ones(len(data))
    # print(data.head())
    return data

def line(data, test) :
    x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'ne', 'nw', 'se', 'sw', 'x0']].to_numpy(dtype=numpy.float64)
    y = data['charges'].to_numpy(dtype=numpy.float64)
    w = numpy.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    xTest = test[['age', 'sex', 'bmi', 'children', 'smoker', 'ne', 'nw', 'se', 'sw', 'x0']].to_numpy(dtype=numpy.float64)
    yTest = xTest.dot(w)
    return yTest


def svr(data, test) :
    x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'ne', 'nw', 'se', 'sw']].to_numpy(dtype=numpy.float64)
    y = data['charges'].to_numpy(dtype=numpy.float64)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    yMean = numpy.mean(y)
    yStd = numpy.std(y)
    y = (y - yMean) / yStd

    C = [1,2,3,4,5]
    epsilon = [0.1,0.2,0.3]
    params = {'C': C, 'epsilon': epsilon}
    model = GridSearchCV(SVR(), params, scoring='r2')
    result = model.fit(x, y)
    means = result.cv_results_['mean_test_score']
    params = result.cv_results_['params']
    for mean,param in zip(means,params):
        print("%f  with:   %r" % (mean,param))
    print('Best params:', model.best_params_)  
    print('Best score:', model.best_score_) 

    # model = SVR(C=2, epsilon=0.1)
    # model.fit(x, y)

    xTest = test[['age', 'sex', 'bmi', 'children', 'smoker', 'ne', 'nw', 'se', 'sw']].to_numpy(dtype=numpy.float64)
    yTest = model.predict(scaler.transform(xTest)) * yStd + yMean
    return yTest

if __name__ == '__main__' :
    mode = int(input())
    data = pandas.read_csv('data/train.csv')
    data = preprocess(data, mode)
    test = pandas.read_csv('data/test_sample.csv')
    test = preprocess(test, mode)
    if (mode == 1) :
        yTest = line(data, test)
        test = pandas.read_csv('data/test_sample.csv')
        test.loc[:, 'charges'] = yTest
        test.to_csv('line_result.csv', index = False)
    elif (mode == 2) :
        yTest = svr(data, test)
        test = pandas.read_csv('data/test_sample.csv')
        test.loc[:, 'charges'] = yTest
        test.to_csv('svr_result.csv', index = False)
    else :
        print("Invalid Input")
    