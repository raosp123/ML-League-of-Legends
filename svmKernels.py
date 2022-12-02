import pandas as pd 
import numpy as np 
import json
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

top_keys = ["turretsLost","teamDamagePercentage","killParticipation","twentyMinionsIn3SecondsCount","visionScore",
              "laneMinionsFirst10Minutes","neutralMinionsKilled","totalDamageDealtToChampions","bountyLevel","goldPerMinute",
             "damagePerMinute","turretTakedowns","damageTakenOnTeamPercentage","kda","champExperience","hadOpenNexus"]

jungle_keys =["turretsLost","totalHealsOnTeammates","killParticipation","consumablesPurchased","teamDamagePercentage",
              "inhibitorsLost","completeSupportQuestInTime","deathsByEnemyChamps","champExperience","bountyLevel","goldPerMinute",
             "damagePerMinute","turretTakedowns","champLevel","kda","totalTimeSpentDead"]

middle_keys =["turretsLost","goldSpent","teamDamagePercentage","killParticipation","totalDamageDealtToChampions","visionScore",
              "deaths","neutralMinionsKilled","visionScorePerMinute","bountyLevel","goldPerMinute",
             "damagePerMinute","turretTakedowns","damageTakenOnTeamPercentage","champLevel","assists"]

bot_keys =["goldSpent", "bountyGold","totalDamageTaken","gameLength","timePlayed","goldEarned","longestTimeSpentLiving",
              "effectiveHealAndShielding","damageDealtToTurrets","damageDealtToBuildings","damageDealtToObjectives","largestCriticalStrike"
             ,"damageSelfMitigated","champExperience","totalHeal","totalDamageDealtToChampions"]

sup_keys =["turretsLost","teamDamagePercentage","killParticipation","goldSpent","visionScore",
              "enemyJungleMonsterKills","skillshotsHit","twentyMinionsIn3SecondsCount","bountyLevel","goldPerMinute",
             "damagePerMinute","turretTakedowns","damageTakenOnTeamPercentage","kda","champExperience","champLevel"]

def svcKernelLimitedFeatures(data, c, gamma, keys):

    y = data.win.astype('int')
    X = data[keys]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,stratify=y)

    model = SVC(C=c,kernel='rbf',gamma=gamma)
    # fit the model
    model.fit(Xtrain, ytrain)
    
    return model, Xtrain, Xtest, ytrain, ytest


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

    fpr1, tpr1, _ = roc_curve(ytest,y_pred)
    plot.plot(fpr1,tpr1)
    fpr2, tpr2, _ = roc_curve(ytest,ydummy)
    plot.title("ROC curve",fontsize=18,pad=20)
    plot.plot(fpr2,tpr2,color='green',linestyle='--')
    plot.xlabel('False positive rate')
    plot.ylabel('True positive rate')
    plot.legend(["Kernalised SVM", "dummy"]) 
    plot.show()

def trainVsTestPredict(data):
    plot.rcParams['figure.constrained_layout.use'] = True

    train_points = []
    test_points = []

    for i in range(10):

        model, Xtrain, Xtest, ytrain, ytest = svcKernelLimitedFeatures(data, 5, 1, top_keys)
        y_pred = model.predict(Xtest)
        print(confusion_matrix(ytest, y_pred))
        c_report_test = classification_report(ytest, y_pred, output_dict=True)["accuracy"]

        test_points.append(c_report_test)

        y_pred2 = model.predict(Xtrain)
        print(confusion_matrix(ytrain, y_pred2))
        c_report_train = classification_report(ytrain, y_pred2, output_dict=True)["accuracy"]

        train_points.append(c_report_train)
        print(f'test accuracy is: {c_report_test}, train accurracy is: {c_report_train}')

    base = np.arange(1, len(train_points)+1,1)

    plot.plot(base, train_points)
    plot.plot(base, test_points)

    plot.title('test vs train accuracy')
    plot.ylabel('accuracy')
    plot.xlabel('crossval iteration')
    plot.legend(['train', 'test'], loc='upper left')
    plot.show()


def gammaCrossVal(data):

    y = data.win.astype('int')
    X = data[['damagePerMinute',
            'bountyLevel', 
            'goldPerMinute',
            'champExperience',
            'visionScorePerMinute',
            'turretTakedowns',
            'kda',
            'hadOpenNexus',
            'turretsLost',
            'visionScore',
            'teamDamagePercentage',
            'totalDamageDealtToChampions',
            'killParticipation',
            'goldSpent',
            'physicalDamageDealtToChampions',
            'jungleCsBefore10Minutes',]]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,stratify=y)

    mean_error=[]
    std_error=[]
    gamma_range = [1, 2, 3, 4, 5]
    for gamma in gamma_range:
      model = SVC(C=1,kernel='rbf',gamma=gamma)
      scores = cross_val_score(model,Xtrain, ytrain, cv=5, scoring='f1')
      mean_error.append(np.array(scores).mean())
      std_error.append(np.array(scores).std())
    plot.rc('font', size=18); plot.rcParams['figure.constrained_layout.use'] = True
    plot.errorbar(gamma_range,mean_error,yerr=std_error,linewidth=3)
    plot.title("penalty parameter vs F1 score",fontsize=18,pad=20)
    plot.xlabel('gamma'); plot.ylabel('F1 Score')
    plot.show()


def cCrossVal(data):

    y = data.win.astype('int')
    X = data[['damagePerMinute',
            'bountyLevel', 
            'goldPerMinute',
            'champExperience',
            'visionScorePerMinute',
            'turretTakedowns',
            'kda',
            'hadOpenNexus',
            'turretsLost',
            'visionScore',
            'teamDamagePercentage',
            'totalDamageDealtToChampions',
            'killParticipation',
            'goldSpent',
            'physicalDamageDealtToChampions',
            'jungleCsBefore10Minutes',]]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,stratify=y)

    mean_error=[]
    std_error=[]
    Ci_range = [1,5,10,20,30]
    for Ci in Ci_range:
      print(f"Starting model {Ci}")
      model = SVC(C=Ci,kernel='rbf',gamma=1)
      scores = cross_val_score(model,Xtrain, ytrain, cv=5, scoring='f1')
      mean_error.append(np.array(scores).mean())
      std_error.append(np.array(scores).std())
    plot.rc('font', size=18); plot.rcParams['figure.constrained_layout.use'] = True
    plot.errorbar(Ci_range,mean_error,yerr=std_error,linewidth=3)
    plot.title("penalty parameter vs F1 score",fontsize=18,pad=20)
    plot.xlabel('Ci'); plot.ylabel('F1 Score')
    plot.show()


    return 1

df = pd.read_csv('league_dataset_60k.csv')
column_keys = df.keys()
scaler = StandardScaler()

#pop the string label and win output before normalisation
teampos = df.pop("teamPosition")
win = df.pop("win")

df = scaler.fit_transform(df)

df = np.column_stack((teampos, df))
df = pd.DataFrame(np.column_stack((df, win)), columns=column_keys)

grouped = df.groupby(df.teamPosition)
data_top = grouped.get_group("TOP")
data_jug= grouped.get_group("JUNGLE")
data_mid = grouped.get_group("MIDDLE")
data_bot = grouped.get_group("BOTTOM")
data_sup = grouped.get_group("UTILITY")


print(f'top size: {data_top.shape}, jungle size: {data_jug.shape}, mid size: {data_mid.shape}, bot size: {data_bot.shape}, sup size: {data_sup.shape},')

print(f'{(df["win"].value_counts()).astype(int)}')

model, Xtrain, Xtest, ytrain, ytest = svcKernelLimitedFeatures(data_sup, 5, 1, sup_keys)

dummyComparison(Xtrain, Xtest, ytrain, ytest, model)
#trainVsTestPredict(data_top)

#gammaCrossVal(df.iloc[0:4000])
#cCrossVal(df.iloc[0:4000])

