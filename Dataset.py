#!/usr/bin/env python
# coding: utf-8


from datasets import load_dataset


# In[1]:


from glob import glob
wavs = glob("{}/**/*.wav".format("AudiosTrimmed"), recursive=True)


# In[2]:


# wavs


# In[124]:


import soundfile as sf
sf.read(wavs[0])


# In[125]:


import soundfile as sf

for file in wavs:
  data, samplerate = sf.read(file)
  sf.write(file, data, 16000, subtype='PCM_16')


# In[126]:


sf.read(wavs[0])


# In[127]:


import pandas as pd

# In[129]:


from datasets import  Audio
from datasets import load_dataset, DatasetDict, Dataset
Data = DatasetDict()


# In[130]:


df = pd.read_csv('text.csv')
df_train=pd.DataFrame()
train_text = []
train_path = []
for i in range(len(df)):
    train_text.append(df.iloc[i]['transcriptions'])
    s = df.iloc[i]['audio_file'].split('/')
    train_path.append("AudiosTrimmed/"+"/".join(s))
df_train['text'] = train_text
df_train['audio'] = train_path
print(df_train.head())


# In[131]:


Data['train'] = Dataset.from_pandas(df_train).cast_column("audio", Audio())


# In[133]:


# Data['train']['audio'][0]


# In[134]:


Data.save_to_disk('Dataset')


# In[107]:


# wavs[10]


# In[ ]:




