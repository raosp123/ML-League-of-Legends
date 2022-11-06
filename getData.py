import requests


puuid = "S01KLVfg1APQYb5xtEoU7QcyKmlpxluPNatYIQE4wmWsallXAw0WBldrlXYMYwrug5PypPYmGtUBjA" # lubis








response = requests.get('https://euw1.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids')
