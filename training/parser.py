import pandas as pd
import numpy as np

df = pd.read_csv('data/tagging_230722.csv')    
dft = pd.read_csv('data/tagging_230722.csv')
taggers = [c for c in dft.columns.values if c!='Title' and c!='year']
df = df.merge(dft, left_on='name',right_on='Title')
df.drop_duplicates('name',inplace=True)
del df['Title']
del dft
df['definite'] = 0
for tagger in taggers:
    df[tagger] = df[tagger].map({'base':1, 'Base':1,'both':0, 'base/both':0, 'center':-1,np.nan:0,'Already Tagged':0}).astype('int')
    df['definite'] = df['definite'] | np.where(df[tagger]==0,0,2**((df[tagger]+1)/2)).astype('int')
#definite data filter
df = df[df['definite']!=3]
del df['definite']
#code label
df['code_label'] = np.sign(df[taggers].sum(axis=1))
#base model
df['label'] = (df['code_label']>=0)+0
df[['name','text','label']].to_csv('tags_base.csv',index=False)
#center model
df['label'] = (df['code_label']<=0)+0
df[['name','text','label']].to_csv('tags_center.csv',index=False)
#3Label mode
df['label'] = df['code_label']+1
df[['name','text','label']].to_csv('tags.csv',index=False)
#split to halves
df['text'] = df['text'].str.replace('"','')
df['sentences'] = df['text'].str.split('.')
df['sentences'] = df['sentences'].apply(lambda x: [e.strip() for e in x])
df['text1'] = df['sentences'].apply(lambda x:". ".join(x[:int(len(x)/2)])+".")
df['copy']=0
tmp_df = df.copy()
tmp_df['copy']=1
tmp_df['text1'] = tmp_df['sentences'].apply(lambda x:".".join(x[int(len(x)/2):]))
df = pd.concat([df,tmp_df],axis=0)
df.reset_index(inplace=True)
df['text'] = df['text1']
df.sort_values(['name'],inplace=True)
del tmp_df
del df['text1']
del df['sentences']
#code label
df['code_label'] = np.sign(df[taggers].sum(axis=1))
#base model
df['label'] = (df['code_label']>=0)+0
df[['name','copy','text','label']].to_csv('tags_base_double.csv',index=False)
#center model
df['label'] = (df['code_label']<=0)+0
df[['name','text','label']].to_csv('tags_center_double.csv',index=False)
#3Label mode
df['label'] = df['code_label']+1
df[['name','text','label']].to_csv('tags_double.csv',index=False)
