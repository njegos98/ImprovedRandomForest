#!/usr/bin/env python
# coding: utf-8

# ## --- 0.0. Libraries Importing ---

# In[1]:


# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from collections import Counter
import random

# datetime
from datetime import datetime

# stats
from scipy.stats import chi2

# set notebook width to 100%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# text coloring
from termcolor import colored


# # 1. Algorithm 

# ## --- 1.1. Decision Tree ---

# In[2]:


### 1. NODE  

class Node:
    
    def __init__(self, feature=None, feature_type=None, thresholds=None, childs=None, gain=None, value=None):
        self.feature = feature
        self.feature_type = feature_type
        self.thresholds = thresholds
        self.childs = childs
        self.gain = gain
        self.value = value
        
    def __str__(self):
        return f'feature = {self.feature}\nfeature_type = {self.feature_type}\nthresholds = {self.thresholds}\ngain = {self.gain}\nvalue = {self.value}'
    
### 2. DECISIONTREE

class DecisionTree:
    
    def __init__(self, splitting_rule='gini', max_depth=2, min_bucket=1, mtry=None, alpha=1, handle_outliers=False, weight_classes=False, uniform_strength=0):
        self.splitting_rule = splitting_rule
        self.max_depth = max_depth
        self.min_bucket = min_bucket 
        self.mtry = mtry
        self.alpha = alpha
        self.handle_outliers = handle_outliers
        self.weight_classes = weight_classes
        self.uniform_strength = uniform_strength
        self.root = None
        self.features = []
    
    ## 2.1. IMPURITY MEASURES

    # 2.1.1. Entropy
    @staticmethod
    def get_entropy(x_weights, y):
        percentages = (pd
                       .DataFrame(data={'Weight':x_weights, 'y':y})
                       .groupby('y')['Weight']
                       .sum()
                       .reset_index()
                       .assign(Weight_Normalized=lambda x: x['Weight'] / x['Weight'].sum())
                       .loc[:, 'Weight_Normalized']
                       .values
                      )
    
        return (percentages*np.log2(1/percentages)).sum()

    # 2.1.2. Gini Index
    @staticmethod
    def get_gini(x_weights, y):
        percentages = (pd
                       .DataFrame(data={'Weight':x_weights, 'y':y})
                       .groupby('y')['Weight']
                       .sum()
                       .reset_index()
                       .assign(Weight_Normalized=lambda x: x['Weight'] / x['Weight'].sum())
                       .loc[:, 'Weight_Normalized']
                       .values
                      )
        
        return 1 - np.power(percentages, 2).sum()

    # 2.1.3. Impurity
    @staticmethod
    def get_impurity(x_weights, y, splitting_rule):
        return (
                DecisionTree.get_entropy(x_weights=x_weights, y=y) if splitting_rule == 'entropy'
                else DecisionTree.get_gini(x_weights=x_weights, y=y)
               )

    # 2.1.4. Impurity Gain
    @staticmethod
    def get_impurity_gain(parent_weights, parent_y, childs_weights, childs_y, splitting_rule):
        parent_impurity = DecisionTree.get_impurity(x_weights=parent_weights,
                                                    y=parent_y,
                                                    splitting_rule=splitting_rule)
    
        percentages = np.array([len(child_y) / len(parent_y) for child_y in childs_y], dtype=np.float64)
        impurities = np.array([DecisionTree.get_impurity(x_weights=x_weights, y=y, splitting_rule=splitting_rule) for x_weights, y in list(zip(childs_weights, childs_y))], dtype=np.float64)
        childs_impurity = (percentages*impurities).sum()
        
        return parent_impurity - childs_impurity
    
    ## 2.2. STATISTICAL TESTING
    
    # 2.2.1. Chi2 
    @staticmethod
    def chi2_test(X, y):
        # observed distribution
        sample_distribution = pd.crosstab(X, y)
    
        x_categories, y_categories = sample_distribution.index.tolist(), sample_distribution.columns.tolist()
    
        # expected distribution
        theoretical_distribution = pd.crosstab(X, y)
        theoretical_distribution['Marginal_x'] = theoretical_distribution.sum(axis=1)
        theoretical_distribution.loc['Marginal_y'] = theoretical_distribution.sum(axis=0)
    
        for x_cat in x_categories:
            for y_cat in y_categories:
                theoretical_distribution.loc[x_cat, y_cat] = (theoretical_distribution.loc[x_cat, 'Marginal_x']*theoretical_distribution.loc['Marginal_y', y_cat]) / (len(y))
            
        theoretical_distribution = (theoretical_distribution
                                    .drop(columns='Marginal_x')
                                    .drop(index='Marginal_y')
                                   )
                
        # dof
        r, s = len(x_categories), len(y_categories)
    
        # Statistics value (with Yates's correction)
        if ((r, s) == (2, 2)) & (len(y) <= 40) & (np.array(theoretical_distribution).min() < 5):
            chi_stat = np.array(((abs(sample_distribution - theoretical_distribution) - 0.5)**2) / theoretical_distribution).sum()     
        else:
            chi_stat = np.array(((sample_distribution - theoretical_distribution)**2) / theoretical_distribution).sum()
        
        # p-value
        p_value = 1 - chi2.cdf(chi_stat, (r-1)*(s-1))
            
        return {'Statistics_value':chi_stat, 'p_value':p_value}
    
    # 2.2.2. Kruskal-Wallis
    @staticmethod
    def kruskal_wallis_test(X, y):
        # sorted dataframe and ranks
        df_rank_per_value = (pd.DataFrame(data={'X':X.values, 'y':y.values})
                             .sort_values(by='X', ascending=True)
                             .assign(Rank=(np.arange(len(y))+1))
                             .groupby('X')['Rank']
                             .mean()
                             .reset_index()
                            )
    
        df_ranks = (pd.DataFrame(data={'X':X.values, 'y':y.values})
                    .sort_values(by='X', ascending=True)
                    .merge(right=df_rank_per_value,
                           how='inner',
                           on='X')
                   )
        
        # df_test
        df_test = pd.DataFrame(df_ranks
                               .groupby('y')['Rank']
                               .agg(['sum', 'count'])
                               .assign(Sum_Of_Squared_Ranks=lambda x: np.array(x['sum'], dtype='float')**2)
                               .assign(Average_Squared_Rank=lambda x: x['Sum_Of_Squared_Ranks'] / x['count'])
                               .reset_index()
                              )    
    
        # dof, h_stat, p_value
        dof = y.nunique() - 1
        h_stat = (12 / ((len(y)*(len(y)+1))))*(df_test['Average_Squared_Rank'].sum()) - 3*(len(y)+1)
        p_value = 1 - chi2.cdf(h_stat, dof) 
    
        return {'Statistics_value':h_stat, 'p_value':p_value}

    # 2.2.3. method to check statistical significance
    @staticmethod
    def statistical_significance(X, y, alpha):
        return (
                (DecisionTree.kruskal_wallis_test(X=X, y=y)['p_value'] < alpha) if DecisionTree.numerical(X)
                else (DecisionTree.chi2_test(X=X, y=y)['p_value'] < alpha)
               )

    # 2.3 OUTLIERS
    @staticmethod
    def drop_outliers(X_df, X, y):
        descriptive_statistics = (X
                                  .describe()
                                  .reset_index()
                                  .set_index('index')
                                 )
    
        q1, q3 = descriptive_statistics.loc['25%', X.name], descriptive_statistics.loc['75%', X.name]
        iqr = (q3 - q1)
        bottom_whisker, upper_whisker = (q1 - 1.5*iqr) if (X.min() < (q1 - 1.5*iqr)) else X.min(), (q3 + 1.5*iqr) if (X.max() > (q3 + 1.5*iqr)) else X.max()
    
        indices_to_drop = (X[(X < bottom_whisker) | (X > upper_whisker)]
                           .index
                           .tolist()
                          )
    
        return ( (X_df
                  .drop(index=indices_to_drop)
                  .reset_index(drop=True)
                 ),
            
                 (X
                  .drop(index=indices_to_drop)
                  .reset_index(drop=True)
                 ),
            
                (y
                 .drop(index=indices_to_drop)
                 .reset_index(drop=True)
                )
              )
    
    ## 2.4. CLASS WEIGHTS
    def get_weights(self, X, y):
        output = ()
        if self.weight_classes:
            data_w = (pd
                      .concat(objs=[X, y], axis=1)
                      .merge(right=(y
                                    .value_counts(normalize=True)
                                    .reset_index()
                                    .assign(Weight=lambda x: (1 - x[y.name]) / (x[y.name]))
                                    .assign(Mean_Weight=lambda x: x['Weight'].mean())
                                    .assign(Diff_Sign=lambda x: np.sign(x['Weight'] - x['Mean_Weight']))
                                    .assign(Weight=lambda x: x['Weight'] - x['Diff_Sign']*np.abs(x['Weight'] - x['Mean_Weight'])*self.uniform_strength)
                                    .drop(columns=[y.name, 'Mean_Weight', 'Diff_Sign'])
                                    .rename(columns={'index': y.name})
                                   ),
                             how='inner', on=y.name
                            )
                      )
                
            X_w = (data_w
                   .drop(columns=y.name)
                   .sample(frac=1)
                   .reset_index()
                  )
    
            y_w = (data_w
                   .loc[X_w['index'], y.name]
                   .reset_index(drop=True)
                  )
               
            output = (X_w.drop(columns='index'), y_w)
        else:
            output = ((X.assign(Weight=1)), y)
            
        return output
                   
    ## 2.5. ALGORITHM
    @staticmethod
    def numerical(X):
        return (('int' in str(X.dtype)) | ('float' in str(X.dtype)))
    
    def min_bucket_check(self, childs_data):
        return np.array(list(map(int, childs_data < self.min_bucket))).sum() == 0   

    def best_split(self, X, y):
        
        best_split, best_gain, columns = {}, -1, X.columns.tolist()
        columns.remove('Weight')
                    
        # features subset
        if self.mtry != None:
            columns = random.sample(columns, self.mtry)
                    
        for col in columns:
            
            # outliers check
            if (DecisionTree.numerical(X[col])) & (self.handle_outliers):
                X_df_current, X_current, y_current = (DecisionTree
                                                      .drop_outliers(X, X[col], y)
                                                     )
            else:
                X_df_current, X_current, y_current = X.copy(), X[col].copy(), y.copy()
            
            # statistical significance
            if DecisionTree.statistical_significance(X_current, y_current, self.alpha) == False:
                continue
                        
            df_parent = pd.concat(objs=[pd.DataFrame(X_df_current), pd.DataFrame(y_current)],
                                  axis=1)
        
            parent_weights = df_parent['Weight'].values
            parent_y = df_parent[y_current.name].values
            
            # numerical predictor
            if DecisionTree.numerical(X_current):
                                
                # run through all unique values (sorted)
                thresholds = X_current.unique()
                thresholds.sort()
                for thr in thresholds:
                    
                    df_left_child = (df_parent[df_parent[col] <= thr]
                                     .copy()
                                     .reset_index(drop=True)
                                    )
                    
                    df_right_child = (df_parent[df_parent[col] > thr]
                                      .copy()
                                      .reset_index(drop=True)
                                     ) 
                    
                    left_child_weights = df_left_child['Weight'].values
                    left_child_y = df_left_child[y_current.name].values
                    
                    right_child_weights = df_right_child['Weight'].values
                    right_child_y = df_right_child[y_current.name].values
                                        
                    gain = (self
                            .get_impurity_gain(parent_weights=parent_weights,
                                               parent_y=parent_y,
                                               childs_weights=[left_child_weights, right_child_weights],
                                               childs_y=[left_child_y, right_child_y],
                                               splitting_rule=self.splitting_rule)
                           )
                    
                    condition_to_split = (
                                            (gain > best_gain) & # GAIN
                                            (self.min_bucket_check(childs_data=np.array([len(df_left_child), len(df_right_child)]))) # MIN_BUCKET
                                         )
                                        
                    if condition_to_split:
                        best_split = {
                            'feature': col,
                            'feature_type': 'numerical',
                            'threshold': thr,
                            'childs': [df_left_child, df_right_child],
                            'gain': gain
                        }
                        best_gain = gain
                                                
            # categorical predictor
            else:
                childs, childs_weights, childs_y, childs_count = [], [], [], []
                
                # run through all unique values
                thresholds = X_current.unique()
                for thr in thresholds:
                    child = (df_parent[df_parent[col] == thr]
                             .reset_index(drop=True)
                             .copy()
                            )
                    child_weight = child['Weight'].values
                    child_y = child[y_current.name].values
                    child_count = len(child)
                    
                    childs.append(child)
                    childs_weights.append(child_weight)
                    childs_y.append(child_y)
                    childs_count.append(child_count)
                    
                gain = (self
                        .get_impurity_gain(parent_weights=parent_weights,
                                           parent_y=parent_y,
                                           childs_weights=childs_weights,
                                           childs_y=childs_y,
                                           splitting_rule=self.splitting_rule)
                       )
                
                condition_to_split = (
                                        (gain > best_gain) & # GAIN
                                        (self.min_bucket_check(childs_data=np.array(childs_count))) # MIN_BUCKET
                                     )
                
                if condition_to_split:
                    best_split = {
                        'feature': col,
                        'feature_type': 'categorical',
                        'threshold': X_current.unique(),
                        'childs': childs,
                        'gain': gain,
                    }
                    best_gain = gain
                    
        return best_split
    
    def build(self, X, y, depth=0):        
        if depth <= self.max_depth:
            best_split = self.best_split(X, y)
            if len(best_split) > 0:
                if (best_split['gain'] > 0):
                    
                    if best_split['feature'] not in self.features:
                        self.features.append(best_split['feature'])
                        
                    child_nodes = []
                    for child in best_split['childs']:
                        child_node = self.build(X=child.drop(columns=y.name),
                                                y=child[y.name],
                                                depth=depth+1)
                        child_nodes.append(child_node)
                
                    return Node(feature=best_split['feature'],
                                feature_type=best_split['feature_type'],
                                thresholds=best_split['threshold'],
                                childs=child_nodes,
                                gain=best_split['gain'])
        
        return Node(
                     value=(pd
                            .concat(objs=[X['Weight'], y], axis=1)
                            .groupby(y.name)['Weight']
                            .sum()
                            .idxmax()
                           )
                   )
    
    def fit(self, X, y):
        X_w, y_w = self.get_weights(X, y)
        self.root = self.build(X_w, y_w)
        
    def predict_instance(self, x, tree):
        if tree.value != None:
            return tree.value
        
        feature_value = x[tree.feature]
        
        # numerical predictor
        if tree.feature_type == 'numerical':
            if feature_value <= tree.thresholds:
                return self.predict_instance(x=x,
                                             tree=tree.childs[0])
            else:
                return self.predict_instance(x=x,
                                             tree=tree.childs[1])
        # categorical predictor
        else:
            return self.predict_instance(x=x,
                                         tree=tree.childs[tree.thresholds.tolist().index(feature_value)])
                
    def predict(self, X):
        return np.array([self.predict_instance(X.loc[row_num].reindex(self.features), self.root) for row_num in X.index.tolist()])
    
    def __str__(self):
        return   f'''
    **********************************************************
                       {colored('DECISION TREE MODEL', 'green', attrs=['bold'])}
        
      ------------------ {colored('Hyperparameters', 'red', attrs=['bold'])} -------------------
        {colored('splitting_rule', 'black', attrs=['bold'])} -> {colored(self.splitting_rule, 'blue', attrs=['bold'])}
        {colored('max_depth', 'black', attrs=['bold'])} -> {colored(self.max_depth, 'blue', attrs=['bold'])}
        {colored('min_bucket', 'black', attrs=['bold'])} -> {colored(self.min_bucket, 'blue', attrs=['bold'])}
        {colored('mtry', 'black', attrs=['bold'])} -> {colored(self.mtry, 'blue', attrs=['bold'])}
        {colored('alpha', 'black', attrs=['bold'])} -> {colored(self.alpha, 'blue', attrs=['bold'])}
        {colored('handle_outliers', 'black', attrs=['bold'])} -> {colored(self.handle_outliers, 'blue', attrs=['bold'])}
        {colored('weight_classes', 'black', attrs=['bold'])} -> {colored(self.weight_classes, 'blue', attrs=['bold'])}
        {colored('uniform_strength', 'black', attrs=['bold'])} -> {colored(self.uniform_strength, 'blue', attrs=['bold'])}
      ------------------------------------------------------
    
    **********************************************************\n
    '''


# ## --- 1.2. Random Forest ---

# In[60]:


class RandomForest:
    
    def __init__(self, number_of_trees=10, sample_size=1, replace=False, splitting_rule='gini', max_depth=2, min_bucket=1, mtry=None, alpha=1, handle_outliers=False, weight_classes=False, uniform_strength=0):
        self.number_of_trees = number_of_trees
        self.sample_size = sample_size
        self.replace = replace
        self.splitting_rule = splitting_rule
        self.max_depth = max_depth
        self.min_bucket = min_bucket
        self.mtry = mtry
        self.alpha = alpha
        self.handle_outliers = handle_outliers
        self.weight_classes = weight_classes
        self.uniform_strength = uniform_strength
        
        # trees (individual decision trees)
        self.trees = []
        
    def sample(self, X, y):
        X_sample = (X
                    .sample(frac=self.sample_size, replace=self.replace)
                    .reset_index()
                   )
        
        y_sample = (y
                    .loc[X_sample['index'].values]
                    .reset_index(drop=True)
                   )
        
        return (X_sample.drop(columns='index'), y_sample)
    
    def fit(self, X, y):        
        if len(self.trees) > 0:
            self.trees = []
        
        print('**********************************************************')
        print('TRAINING STARTED...\n')
        time_per_run = []
        for num_runs in range(self.number_of_trees):
            time_run_started = datetime.now()
            
            # sample X, y
            X_sample, y_sample = self.sample(X, y)
            
            # initialize a tree
            tree = DecisionTree(splitting_rule=self.splitting_rule,
                                max_depth=self.max_depth,
                                min_bucket=self.min_bucket,
                                mtry=self.mtry,
                                alpha=self.alpha,
                                handle_outliers=self.handle_outliers,
                                weight_classes=self.weight_classes, 
                                uniform_strength=self.uniform_strength)
            
            # fit the tree
            tree.fit(X_sample, y_sample)
            
            time_run_finished = datetime.now()
            time_run_took = (time_run_finished-time_run_started).seconds / 60
            time_per_run.append(time_run_took)
            print('TREE ({0}) (took: {1} mins).'.format(num_runs, np.round(time_run_took, 2)))
            
            # append fitted tree to the list of trees
            self.trees.append(tree)
            
        print('\nTRAINING FINISHED!\n')
        print('TRAINING TOOK: {0} mins.'.format(np.round(np.array(time_per_run).sum(), 2)))
        print('**********************************************************\n')

            
    def predict(self, X):
        y = []
        for tree in self.trees:
            y.append(tree.predict(X))
        
        y = np.swapaxes(a=y,
                        axis1=0,
                        axis2=1)
        
        predictions = []
        for preds in y:
            predictions.append(Counter(preds).most_common(1)[0][0])
        
        return predictions
    
    def __str__(self):
        return   f'''
    **********************************************************
                       {colored('RANDOM FOREST MODEL', 'green', attrs=['bold'])}
        
      ------------------ {colored('Hyperparameters', 'red', attrs=['bold'])} -------------------
        {colored('num_trees', 'black', attrs=['bold'])} -> {colored(self.number_of_trees, 'blue', attrs=['bold'])}
        {colored('sample_size', 'black', attrs=['bold'])} -> {colored(self.sample_size, 'blue', attrs=['bold'])}
        {colored('replace', 'black', attrs=['bold'])} -> {colored(self.replace, 'blue', attrs=['bold'])}
        {colored('splitting_rule', 'black', attrs=['bold'])} -> {colored(self.splitting_rule, 'blue', attrs=['bold'])}
        {colored('max_depth', 'black', attrs=['bold'])} -> {colored(self.max_depth, 'blue', attrs=['bold'])}
        {colored('min_bucket', 'black', attrs=['bold'])} -> {colored(self.min_bucket, 'blue', attrs=['bold'])}
        {colored('mtry', 'black', attrs=['bold'])} -> {colored(self.mtry, 'blue', attrs=['bold'])}
        {colored('alpha', 'black', attrs=['bold'])} -> {colored(self.alpha, 'blue', attrs=['bold'])}
        {colored('handle_outliers', 'black', attrs=['bold'])} -> {colored(self.handle_outliers, 'blue', attrs=['bold'])}
        {colored('weight_classes', 'black', attrs=['bold'])} -> {colored(self.weight_classes, 'blue', attrs=['bold'])}
        {colored('uniform_strength', 'black', attrs=['bold'])} -> {colored(self.uniform_strength, 'blue', attrs=['bold'])}
      ------------------------------------------------------
    
    **********************************************************
    '''

