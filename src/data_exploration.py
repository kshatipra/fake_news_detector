import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#default theme
plt.style.use('ggplot')
sns.color_palette("tab10")
sns.set_theme(context = 'notebook' , style='darkgrid', font='sans-serif', font_scale = 1, rc = None)
plt.rcParams['figure.figsize'] = [20, 8]
plt.rcParams.update({'font.size': 15})
plt.rcParams['font.family'] = 'sans-serif'

#read the data
df  = pd.read_csv('src/dataset/fake_or_real_news.csv')


#DataFlair - Get the labels
labels=df.label
labels.head()