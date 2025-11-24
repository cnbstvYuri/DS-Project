import pandas as pd

df = pd.read_csv('data/heart.csv')

print("--- TIRA TEIMA DO OLDPEAK ---")
print("Média de Oldpeak por Target:")
print(df.groupby('target')['oldpeak'].mean())
print("-----------------------------")