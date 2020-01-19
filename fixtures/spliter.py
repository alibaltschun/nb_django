import pandas as pd

df = pd.read_csv("./latih.csv",engine='python')
df = df.dropna()
df = df.reset_index()

train=df.sample(frac=0.8,random_state=42) #random state is a seed value
test=df.drop(train.index)

train[['kalimat','kelas']].to_csv("./train.csv",index=False)
test[['kalimat','kelas']].to_csv("./test.csv",index=False)