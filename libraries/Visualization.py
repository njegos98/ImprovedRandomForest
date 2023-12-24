#!/usr/bin/env python
# coding: utf-8

# ## --- 0.0. Libraries importing ---

# In[1]:


# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('pastel')

# set notebook width to 100%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# text coloring
from termcolor import colored

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


# ## --- 1. Functions ---

# In[1]:


def format_plot(ax, size_axes=14, size_title=15):
    font_title = {"size": size_title, "weight": 600, "name": "monospace"}

    font_axes = {"size": size_axes, "weight": "bold", "name": "monospace"}

    ax.grid(True, linestyle=":", alpha=0.6)
    sns.despine(ax=ax, left=True)

    if ax.get_legend():
        ax.legend(bbox_to_anchor=(1.1, 1))

    ax.set_title(f"\n\n{ax.get_title()}\n", fontdict=font_title)
    ax.set_xlabel(f"\n{ax.get_xlabel()} ➞", fontdict=font_axes)
    ax.set_ylabel(f"{ax.get_ylabel()} ➞\n", fontdict=font_axes)
    
def adjust_plot_size(number_of_categories):
    base_number = 4
    plot_dim = 6
    if number_of_categories > base_number:
        plot_dim = plot_dim + (number_of_categories - base_number)*0.7
    return plot_dim
    
def plot_distribution(df, column):
    # prepare plotting structure
    df_plot = (pd.DataFrame(df[column]
                            .fillna('N/A')
                            .value_counts()
                            .sort_index()
                           )
               .reset_index()
              )
    
    # take adjusted plot dimension
    plot_dim = adjust_plot_size(df_plot.shape[0])
    
    # plot distribution
    _, ax = plt.subplots(1,1,figsize=(plot_dim, plot_dim))
    sns.barplot(x='index',
                y=column,
                data=df_plot,
                ax=ax)
    
    for bar in ax.patches:
        # absolute frequencies
        ax.annotate(format(bar.get_height(), '.0f'),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center',
                    va='center', 
                    size=12, 
                    xytext=(0, 8),
                    textcoords='offset points')
        # relative frequencies
        if (bar.get_height() / df.shape[0])*100 > 3:
            ax.annotate(format((bar.get_height() / df.shape[0])*100, '.2f') + ' %',
                        (bar.get_x() + bar.get_width() / 2, (bar.get_height() / 2) - 0.08*bar.get_height()),
                        ha='center',
                        va='center',
                        size=12, 
                        xytext=(0, 8),
                        textcoords='offset points')
            
    if (len(ax.get_xticklabels()) >= 4):
        ax.set_xticklabels(ax.get_xticklabels(), 
                           rotation=45, 
                           horizontalalignment='right')
        
    ax.set_xlabel(column, size=13)
    ax.set_ylabel('Count', size=13)
    ax.set_title(f'*** Distribution of the variable {column} ***')
     
    format_plot(ax)
    plt.show()

def plot_performances(y_true, y_hat):
    ##### confusion matrix
    _, ax = plt.subplots(1,2,figsize=(12, 6))
    ax1, ax2 = ax[0], ax[1]
    
    # abs counts
    conf_matrix = pd.crosstab(index=y_true,
                              columns=y_hat)
    
    sns.heatmap(data=conf_matrix,
                cmap="YlGnBu",
                annot=True,
                cbar=False,
                fmt=".0f", 
                ax=ax1)
    
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion matrix (absolute)')
    
    # normalized by actual class
    conf_matrix_norm = pd.crosstab(index=y_true,
                                   columns=y_hat,
                                   normalize='index')
    
    sns.heatmap(data=conf_matrix_norm,
                cmap="YlGnBu",
                annot=True,
                cbar=False, 
                fmt=".2f",
                ax=ax2)
    
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion matrix (normalized by class)')
    
    format_plot(ax1)
    format_plot(ax2)
    plt.show()
    
    print(f'''
    {colored('Accuracy', 'black', attrs=['bold'])}: {colored(str(round(accuracy_score(y_true, y_hat), 3)), 'green', attrs=['bold'])}.
    {colored('Precision', 'black', attrs=['bold'])}: {colored(str(round(precision_score(y_true, y_hat), 3)), 'green', attrs=['bold'])}.
    {colored('Recall', 'black', attrs=['bold'])}: {colored(str(round(recall_score(y_true, y_hat), 3)), 'green', attrs=['bold'])}.
    {colored('F1', 'black', attrs=['bold'])}: {colored(str(round(f1_score(y_true, y_hat), 3)), 'green', attrs=['bold'])}.
    ''')
    
def plot_heatmap(df_evaluation, x_label, y_label, title):
    _, ax = plt.subplots(1,1, figsize=(len(df_evaluation.index.tolist())*1.5, len(df_evaluation.columns.tolist())*1.5))
    
    for col in df_evaluation.columns:
        df_evaluation[col] = df_evaluation[col].astype(float)
    
    sns.heatmap(data=df_evaluation,
                cmap="YlGnBu",
                annot=True,
                cbar=False,
                fmt='.3f',
                ax=ax)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    format_plot(ax)
    
    plt.show()
    
def plot_classes_performances(df_c0, df_c1, df_t, x_label, y_label, title_0, title_1, title_t):
    _, ax = plt.subplots(1,3, figsize=(3*len(df_c0.index.tolist())*1.5, len(df_c0.columns.tolist())*1.5))
        
    for col in df_c0.columns.tolist():
        df_c0[col] = df_c0[col].astype(float)
        df_c1[col] = df_c1[col].astype(float)
        df_t[col] = df_t[col].astype(float)
        
    sns.heatmap(data=df_c0,
                cmap="YlGnBu",
                annot=True,
                cbar=False,
                fmt='.3f',
                ax=ax[0])
    
    sns.heatmap(data=df_c1,
                cmap="YlGnBu",
                annot=True,
                cbar=False,
                fmt='.3f',
                ax=ax[1])
    
    sns.heatmap(data=df_t,
                cmap="YlGnBu",
                annot=True,
                cbar=False,
                fmt='.2f',
                ax=ax[2])
    
    ax[0].set_yticklabels(ax[0].get_yticklabels(),
                          rotation=0)
    
    ax[1].set_yticklabels(ax[1].get_yticklabels(),
                          rotation=0)
    
    ax[2].set_yticklabels(ax[2].get_yticklabels(),
                          rotation=0)
    
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    ax[0].set_title(title_0)
    
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)
    ax[1].set_title(title_1)
    
    ax[2].set_xlabel(x_label)
    ax[2].set_ylabel(y_label)
    ax[2].set_title(title_t)
    
    format_plot(ax[0])
    format_plot(ax[1])
    format_plot(ax[2])
    
    plt.show()

