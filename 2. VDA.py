# -*- coding: utf-8 -*-
"""
Created on Sat May  4 05:52:15 2019
Visual Data Analysis in Python Part 1
https://www.kaggle.com/kashnitsky/topic-2-visual-data-analysis-in-python
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg'

# 1. Importing the dataset
df = pd.read_csv('C:/Users/AJ/Documents/Kaggle ML Course/telecom_churn.csv')
df.head()

# 2. Univariate visualization

## 1. Quantitative features

### Histogram
features = ['total day minutes', 'total intl calls']
df[features].hist(figsize=(10,4));

### Density plot
df[features].plot(kind='density', subplots=True, layout=(1, 2), sharex=False, figsize=(10, 4));
# Using seaborn displot to plot histogram and kernel density estimate (KDE)
sns.distplot(df['total intl calls'])

### Box plot
sns.boxplot(x='total intl calls', data=df);

### Violin plot
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
sns.boxplot(data=df['total intl calls'], ax=ax[0]);
sns.violinplot(data=df['total intl calls'], ax=ax[1]);

## 2. Categorical and binary features

### Bar plot (graphical representation of the frequency table)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x='churn', data=df, ax=ax[0]);
sns.countplot(x='customer service calls', data=df, ax=ax[1]);

# 3. Multivariate visualization

## 1. Quantitative vs Quantitative

### 1. Correlation matrix and heatmap
# using correlation matrix to look at the correlations between numerical variables
# separating non-numerical variables
numerical = list(set(df.columns) - set(['state', 'international plan', \
                 'voice mail plan', 'area code', 'churn', 'customer service calls']))
# creating a correlation matrix of numerical variables
corr_matrix = df[numerical].corr()
# using heatmap to visualize the correlation matrix
sns.set(font_scale=1.5) # scale plot text
plt.figure(figsize=(12, 8)) # scale plot size
sns.heatmap(corr_matrix, cmap='Blues_r')
# removing variables that have been calculated directly from number of minutes spent on phone calls
numerical = list(set(numerical) - set(['total day charge', 'total eve charge',\
                 'total night charge', 'total intl charge', 'phone number']))

### 2. Scatter plot
# pyplot scatter
plt.figure(figsize=(12, 8))
plt.scatter(df['total day minutes'], df['total night minutes']);
# seaborn jointplot scatter
sns.jointplot(x='total day minutes', y='total night minutes', data=df, kind='scatter', color='r', height=10)
# seaborn jointplot KDE
sns.jointplot(x='total day minutes', y='total night minutes', data=df, kind='kde', color='g', height=10)
# scatter plot matrix (pairplot)
# `pairplot()` may become very slow with the SVG format
%config InlineBackend.figure_format = 'png'
sns.pairplot(df[numerical]);
# set back to SVG format
%config InlineBackend.figure_format = 'svg'

## 2. Quantitative vs Categorical

### 1. lmplot
sns.lmplot('total day minutes', 'total night minutes', data=df, hue='churn', fit_reg=False);

### 2. Box plot

# appending 'customer service calls' back to numerical features
numerical.append('customer service calls')
# showing box plots for all numerical features
fig, axes = plt.subplots(3, 4, figsize=(10, 7))
for idx, feat in enumerate(numerical):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x='churn', y=feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout();

# comparing box and violin plots for 'total day minutes' feature
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
sns.boxplot(x='churn', y='total day minutes', data=df, ax=axes[0]);
sns.violinplot(x='churn', y='total day minutes', data=df, ax=axes[1]);

# using catplot() to analyze churn with two categorical variables ('total day minutes' amd 'customer service calls')
sns.catplot(x='churn', y='total day minutes', col='customer service calls', data=df[df['customer service calls'] < 8],\
            kind='box', col_wrap=4, height=3, aspect=.8);
            
## 3. Categorical vs Categorical

# checking the distribution of ordinal values of 'customer service calls' feature
df['customer service calls'].value_counts()
# plotting it with hue='churn' using countplot()
sns.countplot(x='customer service calls', hue='churn', data=df);
# plotting churn against bingary features (international paln and voice mail plan)
fig, axes = plt.subplots(1,2, sharey=True, figsize=(10, 4))
sns.countplot(x='international plan', hue='churn', data=df, ax=axes[0]);
sns.countplot(x='voice mail plan', hue='churn', data=df, ax=axes[1]);

# contingency table
# ''churn' and 'state'
pd.crosstab(df['state'], df['churn']).T
# calculate the churn rate for each state
df.groupby(['state'])['churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T

## 4. Whole dataset visualizations

### 1. A naive approach
# using hist plot or pairplot 
df[numerical].plot(kind='hist', bins=100, figsize=(10,6));
sns.pairplot(df[numerical]);

### 2. Dimensionality reduction

### 3. Manifold Learning e.g. t-SNE (t-distributed Stochastic Neighbor Embedding)
# importing libs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# change 'yes' and 'no' values into binary values
X = df.drop(['churn', 'state', 'phone number'], axis=1)
X['international plan'] = X['international plan'].map({'yes':1, 'no':0})
X['voice mail plan'] = X['voice mail plan'].map({'yes':1, 'no':0})
# normalizing the data using StandardScaler()
# subtracting the mean from each variable and divide it by its std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# building a t-SNE representation
%%time
tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)
# plot it!
plt.figure(figsize=(10,8))
plt.scatter(tsne_repr[:, 0], # selecting all rows and column 0
            tsne_repr[:, 1], # selecting all rows and column 1
            alpha=.5); 
# color code the plot!
plt.figure(figsize=(10,8))
plt.scatter(tsne_repr[:, 0], # selecting all rows and column 0
            tsne_repr[:, 1], # selecting all rows and column 1
            c=df['churn'].map({False:'blue', True:'orange'}),
            alpha=.5); 
# coloring the binary features from ;international paln' and 'voice mail plan'
fig, axes = plt.subplots(1,2, sharey=True, figsize=(15,6))
for i, x in enumerate(['international plan', 'voice mail plan']):
    axes[i].scatter(tsne_repr[:, 0], tsne_repr[:, 1], c=df[x].map({'yes':'orange', 'no':'blue'}), alpha=.5);
    axes[i].set_title(x);
    
=============================================================================================================
