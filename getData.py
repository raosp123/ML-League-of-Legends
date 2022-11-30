import requests
import json
import time
import pandas as pd
from keys import API_KEY

MAX_DEPTH = 2 #define how far we recursively acquire data
ROOT_PUUID = "CTcDJgXhdtguxxfjjkqMNUa4fjYI8UDJo6sFgkpEeeXepF03gagf4doUz4rO3cJ-bkz3ht4f03BQKg" #id that the recusion starts from

#matchid = "EUW1_6158939054"

def getPlayerID(playerName):
    response = requests.get(f'https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{playerName}?api_key={API_KEY}')
    responsej = response.json()
    if(response.status_code==403):
        return response.status_code
    

    return responsej


def getMatchIDs(playerID):
    response = requests.get(f'https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{playerID}/ids?api_key={API_KEY}')
    responsej = response.json()
    if(response.status_code==403):
        return response.status_code
    

    return responsej


def getMatchData(matchID):
    response =  requests.get(f'https://europe.api.riotgames.com/lol/match/v5/matches/{matchID}?api_key={API_KEY}')
    responsej = response.json()
    if(response.status_code==403):
        return response.status_code
    

    return responsej

#collectData()
#   parameters:
#       player_ID - the root player id we will use as a starting point
#
#   function:
#       sets up the recursive r_collectData() by initializing the data accumulator for our player data and sets the max_depth
#       returns the output into a json temporarily
#
def collectData(player_ID):

    player_data_Accumulator = []
    max_depth = 0

    r_collectData(player_ID, player_data_Accumulator, max_depth)

    #store for testing
    with open("./ML_League_Data_10.json", "w") as fp:
            json.dump( player_data_Accumulator, fp, indent=4)

# r_collectData()
#       parameters:
#           player_ID - the player ID of the current player we are looking at, start point is provided by collectData()
#           playerIDdataAccumulator - the in place array we are modifying to contain all of our data
#           depth - specifies how much we recursively extract player data. the amount of matches we extract follows len(player_data_Accumulator) = sum(n=0 -> depth) [180x20^n]
#
#       function:
#           1. get the match IDs of our current player
#           2. extract the player IDs of all other players from the root players match history (20 matches)
#           3. call extractPlayerData for all of the players in each match (9 players)
#           4. if we have not reached the intended depth, recursively call this function for the other (20 matches)x(9 players) = 180 players
# 
def r_collectData(player_ID, player_data_Accumulator, depth):

    #Get match data from the current player
    match_IDs = getMatchIDs(player_ID)
    match_data = []

    time.sleep(1)

    if match_IDs!=403:
        for match in match_IDs:
            data = getMatchData(match)
            if(data!=403):
                match_data.append(data)

    #Work on extracted match data
    other_participant_IDs = []
    for match in match_data:
        #ensure we are looking at correct gamemode
        if(("info" in match) and (match["info"]["gameMode"]=="CLASSIC")):

            #Particpants to recursively examine later, remove the current player to prevent duplicates
            other_participant_IDs+= (match["metadata"]["participants"])
            other_participant_IDs.remove(player_ID)

            #Call extractPlayerData on each of the 10 participants in the current match
            for player_data in match["info"]["participants"]:
                extractPlayerData(player_data, player_data_Accumulator)
    
    #print(len(other_participant_IDs))

    #If we have not reached the accepted data accumulation depth, recursively call this function 
    if (depth < 1):
        for player in other_participant_IDs:
            r_collectData(player, player_data_Accumulator, depth+1)
    

# extractPlayerData()
#       parameters:
#           player_data - this is the performance data of one participant from a match
#           playerIDdataAccumulator - the in place array we are modifying to contain all of our data
# 
#       function:
#           1. our function deletes unnecessary keys specified by badKeys.txt
#           2. it then changes boolean values to Integers
#           3. finally it prepends the teamPosition to the start of the array and collects all necessary player data without the keys
#           4. it then appends it to the in place playerIDdataAccumulator array
#
def extractPlayerData(player_data, playerIDdataAccumulator):

    try:
        dataAccumulator = []

        #Delete keys that our model does not use
        bad_keys = open('badKeys.txt')
        Lines = bad_keys.readlines()
        for line in Lines:
            del player_data[line.strip()]

        #Change Boolean values to Integers
        for key in player_data:
            if (player_data[key] == False):
                player_data[key] = 0
            elif (player_data[key] == True):
                player_data[key] = 1

        #Retrieve Team Position key and put at the start of the data
        teamPos = player_data.pop("teamPosition")
        challenges = player_data.pop("challenges")
        temp = {"teamPosition":teamPos}

        namedDatapoints = {**challenges,**player_data}
        namedDatapoints = {**temp, **namedDatapoints}

        #Extract the values from the key/value pairs
        #dataAccumulator.append(namedDatapoints)

        playerIDdataAccumulator.append(namedDatapoints) #all the datapoints from this user
    except Exception as e:
        print(e)



with open('ML_League_Data_1.json', 'r') as f:
  df1 = pd.read_json(f)

with open('ML_League_Data_2.json', 'r') as f:
  df2 = pd.read_json(f)
#df.to_csv('test.csv', index=False)

data = []

for i in range(1,10):
    with open(f'ML_League_Data_{(i)}.json', 'r') as f:
        data.append(pd.read_json(f))
        print(i)

fulldata = pd.concat(data)
fulldata.to_csv('league_dataset_60k.csv', index=False)


#print(getPlayerID("Ekko the Neeko"))

# test = getMatchIDs(ROOT_PUUID)

# print(test)


#extractPlayerData()

#store for testing
# with open("./test.json", "w") as fp:
#         json.dump(playerData, fp, indent=4)




