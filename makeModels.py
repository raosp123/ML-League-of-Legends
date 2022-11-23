import json
import pandas as pd

with open('ML_League_Data_d1.json', 'r') as f:
  df = pd.read_json(f)

df.to_csv('test.csv', index=False)