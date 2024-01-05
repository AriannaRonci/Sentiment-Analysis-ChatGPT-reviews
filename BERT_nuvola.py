import numpy as np
import pandas as pd
import time
import datetime
import gc
import random
import nltk
import re
from tabulate import tabulate
from tqdm import trange
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup

def clean_text(text):
    text = text.lower()

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ",
                  text)  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', text, flags=re.MULTILINE)
    text = re.sub(r"https", "", text)

    html = re.compile(r'<.*?>')

    text = html.sub(r'', text)  # Removing html tags

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')  # Removing punctuations

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    text = " ".join(text)  # removing stopwords

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Removing emojis

    return text

## Shuffle Data
def shuffle(df, n=3, axis=0):
    df = df.copy()
    random_states = [2,42,4]
    for i in range(n):
        df = df.sample(frac=1,random_state=random_states[i]) # mischio il dataframe
    return df


df = pd.read_csv("data/file.csv", usecols=['tweets', 'labels'])

Counter(df.labels)

#mapping labels
df['labels'] = df.labels.map({'neutral':0,'good':1,'bad':2})

text = []
for d in df['tweets']:
  text.append(d.replace("\\n",""))

df['tweets'] = text

df = df.drop_duplicates(subset=['tweets'])

new_df = shuffle(df)
new_df


nltk.download('stopwords')
sw = stopwords.words('english')

df['tweets'] = df['tweets'].apply(lambda x: clean_text(x))

split_idx = int(len(df)*0.8)

train_df = new_df.iloc[:split_idx,:]
test_df = new_df.iloc[split_idx:,:]
print('train lenght:',len(train_df))
print(train_df.groupby(['labels'])['tweets'].count())
print('test lenght:',len(test_df))
print(test_df.groupby(['labels'])['tweets'].count())


mask = np.array(Image.open("data/nuvola.png"))

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10, mask=mask, collocations=False).generate(' '.join(df['tweets']))

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.savefig("wordcloud.png")
plt.show()