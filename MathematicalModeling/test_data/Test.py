import pandas as pd

df = pd.DataFrame({'A': ['1', '2'], 'B': ['4', '5']})
print(df)

df.loc[0,'A'] = '999'
print(df)