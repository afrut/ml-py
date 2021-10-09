#exec(open('main.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import math
import pandas as pd
import sklearn.model_selection as sms
import sklearn.pipeline as pipeline
import sklearn.preprocessing as pp
import sklearn.metrics as metrics
import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as gpk
import sklearn.linear_model as slm
import sklearn.naive_bayes as nb
import sklearn.neighbors as neighbors
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.discriminant_analysis as da
import sklearn.ensemble as ensemble
from sklearn.experimental import enable_hist_gradient_boosting
import sklearn.base as sb

class CustomTransformer(sb.BaseEstimator, sb.TransformerMixin):
    def __init__(self, idxNonnum):
        self._fit = False
        self._idxNonnum = idxNonnum
    def fit(self, X, y = None):
        if(not(self._fit)):
            self._fit = True

        return self
    def transform(self, X, y = None):
        return (self.m * X) + self.b

def countNan(df: pd.DataFrame):
    idx = df.isna().any(axis = 1)
    cntMissing = np.count_nonzero(idx)
    N = len(df)
    pctMissing = cntMissing / N
    return (cntMissing, N, pctMissing, idx)

def impute(df: pd.DataFrame):
    tpl = countNan(df)
    df = df.loc[np.logical_not(tpl[3]), :]
    return df

def loadData():
    df = pd.read_csv('.\\ML-MATT-CompetitionQT1920_train.csv')
    dfTest = pd.read_csv('.\\ML-MATT-CompetitionQT1920_test.csv')
    return (df, dfTest)

def preprocData(df, dfTest):
    # TODO: Drop rows for now. Re-visit this.
    df = impute(df)
    dfTest = impute(dfTest)

    # Check that there are no missing values.
    assert(np.all(np.logical_not(df.isna()))), 'Nan values present'
    assert(np.all(np.logical_not(dfTest.isna()))), 'Nan values present'

    # NOTE: maxUE_UL+DL has type object but should be numerical according to problem
    # statement.
    types = {'maxUE_UL+DL': np.float64, 'Time': str, 'CellName': str}
    df = df.astype(types)
    dfTest = dfTest.astype(types)

    # NOTE: Each sample specifies a CellName and a target value. From
    # exploratory data analysis, there are 33 unique CellNames all with roughly
    # the same number of samples. This dataset can be viewed as being 33
    # datasets - one for each CellName. An option would be to build a separate
    # model for each cell. 

    return (df, dfTest)


if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # ----------------------------------------
    # Data loading and formatting
    # ----------------------------------------
    df, dfTest = loadData()
    df, dfTest = preprocData(df, dfTest)

    # ----------------------------------------
    # Constants
    # ----------------------------------------
    np.set_printoptions(precision = 4, suppress = True)
    pd.options.display.float_format = '{:10,.4f}'.format
    seed = 29

    # ----------------------------------------
    # Specify variables and target.
    # ----------------------------------------
    ycols = set(['Unusual'])
    xcols = set(df.columns) - set(ycols)

    # ----------------------------------------
    # Identify numeric and non-numeric data.
    # ----------------------------------------
    numerics = set(df.select_dtypes([np.number]).columns) - ycols
    nonnumerics = xcols - numerics
    idxnumerics = [xcols.index(col) for col in numerics]
    idxnonnumerics = [xcols.index(col) for col in nonnumerics]
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)

    # ----------------------------------------
    # Validation set
    # ----------------------------------------
    validationSize = 0.2
    Xtrain, Xvalid, ytrain, yvalid = sms.train_test_split(X, y
        ,test_size = validationSize
        ,random_state = seed)

    # ----------------------------------------
    # specify cross-validation
    # ----------------------------------------
    k = 10                                                                   # number of folds
    cvsplitter = sms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    
    # One-hot encoder.
    encOhe = pp.OneHotEncoder(sparse = False, drop = 'first')
    encOhe.fit(Xtrain)
    
    # ----------------------------------------
    # Try different piplines
    # - base model
    # - standardized/normalized/min-max-scaled
    # - one-hot encoding
    # - feature selection pipeline
    # - tuning pipeline
    # ----------------------------------------
    # define estimator parameters
    # format: models[name] = (constructor, constructor_args, hyperparameter_grid)
    # TODO: comment/uncomment as needed
    models = dict()
    models['GPC'] = (gp.GaussianProcessClassifier, {'kernel': 1.0 *gpk.RBF(1.0)}, {}) # this is kind of slow
    models['LR'] = (slm.LogisticRegression, {'max_iter': 1000}, {})
    models['PAC'] = (slm.PassiveAggressiveClassifier, {}, {})
    models['PERCPT'] = (slm.Perceptron, {}, {})
    models['RIDGE'] = (slm.RidgeClassifier, {}, {})
    models['SGD'] = (slm.SGDClassifier, {}, {})
    models['BernNB'] = (nb.BernoulliNB, {}, {})
    models['CatNB'] = (nb.CategoricalNB, {}, {}) # look into this further, does not allow negative values
    models['CompNB'] = (nb.ComplementNB, {}, {}) # does not allow negative values
    models['GaussNB'] = (nb.GaussianNB, {}, {})
    models['MultinNB'] = (nb.MultinomialNB, {}, {}) # does not allow negative values
    models['KNN'] = (neighbors.KNeighborsClassifier, {}, {})
    models['RNN'] = (neighbors.RadiusNeighborsClassifier, {'radius': 10}, {})
    models['MLP'] = (nn.MLPClassifier, {'max_iter': 10000}, {})
    models['LinearSVC'] = (svm.LinearSVC, {'max_iter': 10000}, {})
    models['NuSVC'] = (svm.NuSVC, {}, {})
    models['SVC'] = (svm.SVC, {}, {})
    models['TREE'] = (tree.DecisionTreeClassifier, {}, {})
    models['EXTREE'] = (tree.ExtraTreeClassifier, {}, {})
    models['QDA'] = (da.QuadraticDiscriminantAnalysis, {}, {})
    models['LDA'] = (da.LinearDiscriminantAnalysis, {}, {})
    models['BAGTREE'] = (ensemble.BaggingClassifier, {'random_state': seed, 'base_estimator': tree.DecisionTreeClassifier(), 'n_estimators': 30}, {})
    models['ET'] = (ensemble.ExtraTreesClassifier, {}, {})
    models['ADA'] = (ensemble.AdaBoostClassifier, {}, {})
    models['GBM'] = (ensemble.GradientBoostingClassifier, {}, {})
    models['RF'] = (ensemble.RandomForestClassifier, {}, {})
    models['HISTGBM'] = (ensemble.HistGradientBoostingClassifier, {}, {})

    # ----------------------------------------
    # Pipeline definition
    # ----------------------------------------
    pipelines = dict()
    print('Pipeline creation:')
    for entry in models.items():
        name = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        params = entry[1][2]
        print('    Creating pipeline for {0: <16} - '.format(name))

        # Create pipeline with standard-scaling, one-hot encoding, model and
        # hyperparameter tuning.
        estimators = []
        estimators.append(('OneHotEncoder', encOhe))
        estimators.append(('StandardScaler', pp.StandardScaler()))
        # estimators.append((name, model(**args)))
        ppl = pipeline.Pipeline(estimators)
        # if len(params) > 0:
        #     param_grid = dict()
        #     for tpl in params.items():
        #         param_grid[name + '__' + tpl[0]] = tpl[1]
        #     ppl = sms.GridSearchCV(estimator = ppl, param_grid = param_grid)
        # pipelines[name] = ppl
        # ppl.fit(Xtrain)
    print('')

    # create voting and stacking classifiers
    # TODO: specify estimators for voting classifier
    # estimators = list()
    # estimators.append(('LR', pipelines['LR']))
    # estimators.append(('ADA', pipelines['ADA']))
    # estimators.append(('GBM', pipelines['GBM']))
    # estimators.append(('RF', pipelines['RF']))
    # estimators.append(('ScaledKNN', pipelines['ScaledKNN']))
    # estimators.append(('RIDGE', pipelines['RIDGE']))
    # estimators.append(('SVC', pipelines['SVC']))
    # pipelines['VOTE'] = pipeline.Pipeline([('model', ensemble.VotingClassifier(estimators = estimators))])
    # pipelines['STACK'] = pipeline.Pipeline([('model', ensemble.StackingClassifier(estimators = estimators))])

    # # ----------------------------------------
    # # pipeline fitting and scoring
    # # ----------------------------------------
    # print('Pipleine fitting and scoring progress: name - mean accuracy - std accuracy')
    # scoring = 'neg_mean_absolute_error'
    # pipelinenames = list()
    # scores = list()
    # for entry in pipelines.items():
    #     name = entry[0]
    #     print('    {0:<20}'.format(name), end = '')
    #     ppl = entry[1]
    #     score = -1 * sms.cross_val_score(ppl, Xtrain, ytrain, cv = cvsplitter, scoring = scoring)
    #     scores.append(score)
    #     pipelinenames.append(entry[0])
    #     print('{0:.4f} - {1:.4f}'.format(np.mean(score), np.std(score, ddof = 1)))
    # print('')

    # # boxplot of results
    # plt.close('all')
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.boxplot(scores)
    # ax.set_xticklabels(pipelinenames)
    # ax.set_xlabel('Algorithm')
    # ax.set_ylabel('Mean Absolute Error')
    # ax.set_title('Mean Absolute Error of Different Algorithms')

    # # format every xticklabel
    # for ticklabel in ax.get_xticklabels():
    #     ticklabel.set_horizontalalignment('right')  # center, right, left
    #     ticklabel.set_rotation_mode('anchor')       # None or anchor
    #     ticklabel.set_rotation(60)                  # angle of rotation
    #     ticklabel.set_fontsize(12)                  # float

    # fig.tight_layout()
    # plt.show()
    # plt.close('all')

    # # table of results for cross validation
    # arrscores = np.array(scores)
    # dfScores = pd.DataFrame(index = pipelinenames)
    # dfScores['mean'] = np.mean(arrscores, axis = 1)
    # dfScores['std'] = np.std(arrscores, ddof = 1, axis = 1)
    # print('Mean and standard deviation of MAE for different algorithms:')
    # print(dfScores.sort_values(by = ['mean']))
    # print('')

    # # table of results on validation set/holdout set
    # scores = list()
    # pipelinenames = list()
    # scorer = metrics.get_scorer(scoring)
    # print('Validation set scoring:')
    # for entry in pipelines.items():
    #     name = entry[0]
    #     print('    {0:<20}'.format(name), end = '')
    #     ppl = entry[1]
    #     ppl.fit(Xtrain, ytrain)
    #     score = -1 * scorer(ppl, Xvalid, yvalid)
    #     scores.append(score)
    #     pipelinenames.append(entry[0])
    #     print('{0:.4f}'.format(score))
    # print('')

    # # table of results for validation set/holdout set
    # arrscoresvalid = np.array(scores)
    # dfScoresValid = pd.DataFrame(index = pipelinenames)
    # dfScoresValid['mae'] = arrscoresvalid
    # print('MAE for different algorithms:')
    # print(dfScoresValid.sort_values(by = ['mae']))
    # print('')

    # # table of results on entire dataset
    # scores = list()
    # pipelinenames = list()
    # scorer = metrics.get_scorer(scoring)
    # print('All data scoring:')
    # for entry in pipelines.items():
    #     name = entry[0]
    #     print('    {0:<20}'.format(name), end = '')
    #     ppl = entry[1]
    #     ppl.fit(X, y)
    #     score = -1 * scorer(ppl, X, y)
    #     scores.append(score)
    #     pipelinenames.append(entry[0])
    #     print('{0:.4f}'.format(score))
    # print('')

    # # table of results on entire dataset
    # arrscoresall = np.array(scores)
    # dfScoresAll = pd.DataFrame(index = pipelinenames)
    # dfScoresAll['mae'] = arrscoresall
    # print('MAE for different algorithms:')
    # print(dfScoresAll.sort_values(by = ['mae']))
    # print('')