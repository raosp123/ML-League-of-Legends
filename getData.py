import requests
import json
from keys import API_KEY

MAX_DEPTH = 2 #define how far we recursively acquire data
ROOT_PUUID = "kmDsoNSAJDQ38U07YYem3xaUmPeFht0EouZ1h46WhvRYd8lJ4cVntEaSlrixvSTBl0SPnQLaM6pcpw" #id that the recusion starts from

#matchid = "EUW1_6158939054"

def getPlayerID(playerName):
    response = requests.get(f'https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{playerName}?api_key={API_KEY}')
    print(response.json())
    return response


def getMatchIDs(playerID):
    response = requests.get(f'https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{playerID}/ids?api_key={API_KEY}')
    # with open("./matchIDs.json", "w") as fp:
    #     json.dump(response.json(), fp, indent=4)
    return response


def getMatchData(matchID):
    response =  requests.get(f'https://europe.api.riotgames.com/lol/match/v5/matches/{matchID}?api_key={API_KEY}')
    # with open("./matchData.json", "w") as fp:
    #         json.dump(response.json(), fp, indent=4)
    return response


def collectData(playerID):

    #-
    # 1. get match data
    # 2
    # 
    # 
    # -#

    dataAccumulator = []
    depth = 0
    # finalData = {"dataPoints":data}
    # finalData["dataPoints"].append({"test":1, "test2":2 })


    r_collectData(dataAccumulator)

def r_collectData(playerIDdataAccumulator, depth):

    #format the match data of the current user
    
        #append the relevant data

    if (depth <5):
        #get the puuids of everyone in the match
        #recursively call this function
        print(1)
    
    return playerIDdataAccumulator


def dataFormatByKeys(rawData):

    dataAccumulator = []

    # with open('exampleParticipant.json') as json_file:
    #     rawData = json.load(json_file)

    #Delete keys that our model does not use
    bad_keys = open('badKeys.txt')
    Lines = bad_keys.readlines()
    for line in Lines:
        del rawData[line.strip()]

    #Change Boolean values to Integers
    for key in rawData:
        if (rawData[key] == False):
            rawData[key] = 0
        elif (rawData[key] == True):
            rawData[key] = 1

    #Retrieve Team Position key and put at the start of the data
    teamPos = rawData.pop("teamPosition")
    challenges = rawData.pop("challenges")
    temp = {"teamPosition":teamPos}

    namedDatapoints = {**challenges,**rawData}
    namedDatapoints = {**temp, **namedDatapoints}

    #Extract the values from the key/value pairs
    for key in namedDatapoints:
        dataAccumulator.append(namedDatapoints[key])

        ##this was used for testing to store data for later viewing
    # with open("./singledatapoint.json", "w") as fp:
    #         json.dump(dataAccumulator, fp, indent=4)

    # with open("./namedDatapoint.json", "w") as fp:
    #         json.dump(namedDatapoints, fp, indent=4)

    return dataAccumulator #all the datapoints from this user




#store for testing
# with open("./test.json", "w") as fp:
#         json.dump(rawData, fp, indent=4)




