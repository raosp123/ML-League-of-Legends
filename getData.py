import requests

api_key = 'RGAPI-ae3f51f8-1636-4c4a-b00d-33d4e93a5444'
puuid = "S01KLVfg1APQYb5xtEoU7QcyKmlpxluPNatYIQE4wmWsallXAw0WBldrlXYMYwrug5PypPYmGtUBjA" # lubis








response = requests.get('https://euw1.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids')