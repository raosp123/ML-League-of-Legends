# ML-League-of-Legends
Machine Learning project to predict whether a player/team won a game based on each players performance, and then a report to examine how the parameter weights for each feature differs based on position in the team played by the player to identify what performance metric each role should strive to maximise when playing the game.

The files are:

  getData.py:
          -used to acquire all data by making calls to the RIOT-GAMES API
          -recursively go through player data, by manually inputting a starting player's ID (statically set in the file as a professional player), and look at this                  player's match history, where we then recursively do the same for each player in this persons match.
          
  namedDatapoint.json:
          -contains the named keys and our features for one example datapoint
          
  singleDatapoint.json:
          -contains the features without their names (used to convert to csv and train model later)
          
  
