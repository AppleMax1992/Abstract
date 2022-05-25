from operator import index
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
train = pd.read_json('src/train.json')
test = pd.read_json('src/dev.json')
train,val  = train_test_split(train,test_size=0.2,random_state=1)
train.to_csv('src/train.csv',columns=['id','title','content'], encoding = 'utf_8_sig')
val.to_csv('src/val.csv',encoding = 'utf_8_sig')
test.to_csv('src/test.csv',encoding = 'utf_8_sig')

