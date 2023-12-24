#!/usr/bin/env python
# coding: utf-8

# ## --- 0.0. Libraires importing ---

# In[1]:


# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# set notebook width to 100%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # 1. Functions

# In[2]:


def get_categorical_column_distribution(df, column, values_order=None):
    data_to_display = (pd
                       .DataFrame(data=(
                                               df
                                               .loc[:, column]
                                               .value_counts(dropna=False)
                                       )
                                   )
                       .reset_index()
                       .rename(columns={
                                           column: 'Count',
                                           'index': column
                                       }
                               )
                       .assign(Proportion=lambda x: [str(x) + ' %' for x in np.round((x['Count'] / x['Count'].sum())*100, 2)])
                       .set_index(column)
                      )
    
    return (data_to_display.loc[values_order]) if (values_order != None) else data_to_display

def get_numerical_column_distribution(df, column):
    data_to_display = (pd
                       .DataFrame(
                                     df
                                     .loc[:, column]
                                     .describe()
                                     .round(2)
                                     .reset_index()
                                     .rename(columns={
                                                         'index': 'Statistics',
                                                         column: f'Value ({column})'
                                                     }
                                            )
                                     .set_index('Statistics')
                                 )
                      )
    
    return data_to_display

