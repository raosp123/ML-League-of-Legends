import requests
import json
from keys import API_KEY

puuid = "kmDsoNSAJDQ38U07YYem3xaUmPeFht0EouZ1h46WhvRYd8lJ4cVntEaSlrixvSTBl0SPnQLaM6pcpw" # mine
matchid = "EUW1_6158939054"

def getPlayerID(playerName):
    response = requests.get(f'https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{playerName}?api_key={API_KEY}')
    print(response.json())


def getMatchIDs(playerID):

    response = requests.get(f'https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{playerID}/ids?api_key={API_KEY}')
   
    with open("./matchIDs.json", "w") as fp:
        json.dump(response.json(), fp, indent=4)


def getMatchData(matchID):

    response =  requests.get(f'https://europe.api.riotgames.com/lol/match/v5/matches/{matchID}?api_key={API_KEY}')
    with open("./matchData.json", "w") as fp:
            json.dump(response.json(), fp, indent=4)



#getMatchIDs(puuid)
getMatchData(matchid)

#getPlayerID('giantyoghurt')
