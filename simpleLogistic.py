import pandas as pd 
import numpy as np 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve

##parent function that runs our model fully


def logisticRegFullFeatures(data, c):

    y = data.win
    X = data.drop(["teamPosition", "win"], axis=1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,stratify=y)

    model = LogisticRegression(penalty='l2',C=c,solver='lbfgs')
    # fit the model
    model.fit(X, y)
    
    return model, Xtrain, Xtest, ytrain, ytest

def logisticRegLimitedFeatures(data, c):
    y=data.win
    X = data[["goldSpent", "gameLength","timePlayed","bountyGold","goldEarned","champExperience",
              "summonerLevel","damageDealtToTurrets","damageDealtToBuildings","damageDealtToObjectives"]]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,stratify=y)

    model = LogisticRegression(penalty='l2',C=c,solver='lbfgs')
    # fit the model
    model.fit(X, y)
    
    return model, Xtrain, Xtest, ytrain, ytest

def featureImport(model, X):

    importance = model.coef_[0]
    feas = pd.DataFrame()
    feas['features'] = X.keys()
    feas['importance'] = importance
    feas = feas.sort_values(by='importance')

    feas1 = feas[0:95]
    feas2 = feas[95:191]

    plot.rc('font', size=10);
    fig, axes = plot.subplots(figsize=(11,17))
    axes.barh(np.arange(feas1.shape[0]), feas1.importance.values, alpha=0.8,color='orange')
    axes.set_title("feature importance")
    axes.set_yticks(np.arange(feas1.shape[0]))
    axes.set_yticklabels(feas1.features.values, rotation='horizontal')

    plot.show()

    fig, axes = plot.subplots(figsize=(11,17))
    axes.barh(np.arange(feas2.shape[0]), feas2.importance.values, alpha=0.8,color='orange')
    axes.set_title("feature importance")
    axes.set_yticks(np.arange(feas2.shape[0]))
    axes.set_yticklabels(feas2.features.values, rotation='horizontal')
    plot.show()

def dummyComparison(Xtrain, Xtest, ytrain, ytest, comparison_model):

    #compared model
    y_pred = comparison_model.predict(Xtest)
    print(confusion_matrix(ytest, y_pred))
    print(classification_report(ytest, y_pred))

    #dummy classifier
    dummy = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
    ydummy = dummy.predict(Xtest)
    print(confusion_matrix(ytest, ydummy))
    print(classification_report(ytest, ydummy))

    fpr, tpr, _ = roc_curve(ytest,comparison_model.decision_function(Xtest))
    plot.plot(fpr,tpr)
    fpr1, tpr1, _ = roc_curve(ytest,y_pred)
    plot.plot(fpr1,tpr1)
    fpr2, tpr2, _ = roc_curve(ytest,ydummy)
    plot.title("ROC curve",fontsize=18,pad=20)
    plot.plot(fpr2,tpr2,color='green',linestyle='--')
    plot.xlabel('False positive rate')
    plot.ylabel('True positive rate')
    plot.legend(["logistic", "dummy"]) 
    plot.show()

    


def logisticCrossVal(data):

    y = data.win
    X = data[["goldSpent", "gameLength","timePlayed","bountyGold","goldEarned","champExperience",
              "summonerLevel","damageDealtToTurrets","damageDealtToBuildings","damageDealtToObjectives"]]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,stratify=y)

    mean_error=[]
    std_error=[]
    Ci_range = [0.01, 0.1, 1, 5, 10, 25, 50, 100]
    for Ci in Ci_range:
      model = LogisticRegression(C=Ci,penalty='l2');
      scores = cross_val_score(model,Xtrain, ytrain, cv=5, scoring='f1')
      mean_error.append(np.array(scores).mean())
      std_error.append(np.array(scores).std())
    plot.rc('font', size=18); plot.rcParams['figure.constrained_layout.use'] = True
    plot.errorbar(Ci_range,mean_error,yerr=std_error,linewidth=3)
    plot.title("penalty parameter vs F1 score",fontsize=18,pad=20)
    plot.xlabel('Ci'); plot.ylabel('F1 Score')
    plot.show()

df = pd.read_csv('ml_data.csv')
grouped = df.groupby(df.teamPosition)
data_top = grouped.get_group("TOP")
data_jug= grouped.get_group("JUNGLE")
data_mid = grouped.get_group("MIDDLE")
data_bot = grouped.get_group("BOTTOM")
data_sup = grouped.get_group("UTILITY")

print(data_top["win"].value_counts())

#the data you input here will be used for the model
model, Xtrain, Xtest, ytrain, ytest = logisticRegFullFeatures(data_top, 1)

#call the other functions you want to use here

#featureImport(model, Xtrain)
logisticCrossVal(data_top)
dummyComparison(Xtrain, Xtest, ytrain, ytest, model)


