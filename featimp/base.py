''' 
Author  :   Hasan Basri Akçay 
linkedin:   <https://www.linkedin.com/in/hasan-basri-akcay/> 
medium  :   <https://medium.com/@hasan.basri.akcay>
kaggle  :   <https://www.kaggle.com/hasanbasriakcay>
'''

import gc
import pandas as pd
import numpy as np
from regex import R
import scipy.stats as ss
import time
import random

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import seaborn as sns


np.seterr(invalid='ignore')
cm = sns.color_palette("coolwarm", as_cmap=True)


def get_feature_importances(data=None, num_features=[], cat_features=[], bool_features=[], target='target', group='group', method=[], fold_type='kf', nfold=10, task='clf_multiable', 
                            random_state=0, ml_model_name='LGBM', ml_early_stopping_rounds=100, pi_score=None, pi_model_base=None, pi_n_repeats=5, rank_model=StandardScaler()):
    '''
    Calculates the feature importances by using multiple methods and ranking them.

    Args:
        data (DataFrame): DataFrame that holds all values.
        num_features (list): Numeric features in the data.
        cat_features (list): Categorical features in the data.
        target (str): The feature name that is tried to predict.
        group (str): Group name for gkf type fold.
        method (str, list): Method that is selected for use. Only supports 'corr', 'chi2_crosstab', 'chi2', 'anova', 'mi', 'ml', 'pi', 'all'.
        fold_type (str): Fold type for permutation importance and ML importance.
        nfold (int): Fold number for permutation importance and ML importance.
        task (str): Problem type. Only supports classification and regression. Selections are 'reg', 'clf_binary' and 'clf_multiable'.
        random_state (int): Fixing the randomness. 
        ml_model_name (str): ML name for ML importances. Only supports 'LGBM', 'CATBOOST'.
        ml_early_stopping_rounds (int): Early stopping the fitting for ML importance.
        pi_score (str): Score name for permutation importance.
        pi_model_base (sklearn prediction model): ML model for permutation importance.
        pi_n_repeats (int): Repeat number for permutation importance.
        rank_model (sklearn preprocess model): Scaler model for the sum of the all feature importance.

    Returns:
        DataFrame
    '''
    def get_fi_by_name(data_inside, method_name, model_base, score):
        if method_name == 'corr':
            fi_df = get_corr_importances(data=data_inside, num_features=num_features+bool_features, target=target)
        elif method_name == 'chi2_crosstab':
            fi_df = get_chi2_crosstab_importances(data=data_inside, cat_features=cat_features+bool_features, target=target)
        elif method_name == 'chi2':
            fi_df = get_chi2_importances(data=data_inside, features=cat_features+bool_features, target=target)
        elif method_name == 'anova':
            if task in ['clf_multiable', 'clf_binary']:
                fi_df = get_anova_importances(data=data_inside, features=num_features+bool_features, target=target)
            else:
                fi_df = get_anova_importances(data=data_inside, features=cat_features+bool_features, target=target)
        elif method_name == 'mi':
            print(data_inside[cat_features+bool_features+[target]])
            fi_df = get_mutual_info_importances(data=data_inside, features=cat_features+bool_features, target=target, random_state=random_state)
            print(fi_df)
        elif method_name == 'ml':
            fi_df = get_ml_importances(data=data_inside, num_features=num_features, cat_features=cat_features, bool_features=bool_features, target=target, fold_type=fold_type, 
                                        group=group, nfold=nfold, model_name=ml_model_name, task=task, random_state=random_state, early_stopping_rounds=ml_early_stopping_rounds)
        elif method_name == 'pi':
            fi_df = get_permutation_importances(data=data_inside, features=num_features+cat_features+bool_features, target=target, fold_type=fold_type, group=group, nfold=nfold, 
                                                score=score, model_base=model_base, random_state=random_state, n_repeats=pi_n_repeats)
        elif method_name == 'all':
            if task in ['clf_multiable', 'clf_binary']:
                if task == 'clf_binary':
                    start_time = time.time()
                    corr_df = get_corr_importances(data=data_inside, num_features=num_features+bool_features, target=target)
                    corr_df = corr_df.abs()
                    corr_time = time.time() - start_time
                else:
                    corr_df = pd.DataFrame(columns=['Corr'])
                    corr_time = 0.0
                start_time = time.time()
                cat_corr_df = get_chi2_crosstab_importances(data=data_inside, cat_features=cat_features+bool_features, target=target)
                cat_corr_time = time.time() - start_time
                start_time = time.time()
                chi2_df = get_chi2_importances(data=data_inside, features=cat_features+bool_features, target=target)
                chi2_time = time.time() - start_time
                start_time = time.time()
                anova_df = get_anova_importances(data=data_inside, features=num_features+bool_features, target=target)
                anova_time = time.time() - start_time
                start_time = time.time()
                mi_scores_df = get_mutual_info_importances(data=data_inside, features=cat_features+bool_features, target=target, random_state=random_state)
                mi_time = time.time() - start_time
                start_time = time.time()
                ml_importance_df = get_ml_importances(data=data_inside, num_features=num_features, cat_features=cat_features, bool_features=bool_features, target=target, 
                                        fold_type=fold_type, group=group, nfold=nfold, model_name=ml_model_name, task=task, random_state=random_state, 
                                        early_stopping_rounds=ml_early_stopping_rounds)
                ml_time = time.time() - start_time
                start_time = time.time()
                pi_scores_df = get_permutation_importances(data=data_inside, features=num_features+cat_features+bool_features, target=target, fold_type=fold_type, group=group, 
                                                nfold=nfold, score=score, model_base=model_base, random_state=random_state, n_repeats=pi_n_repeats)
                pi_time = time.time() - start_time
                
                fi_df = pd.DataFrame()
                fi_df = fi_df.merge(corr_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(cat_corr_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(chi2_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(anova_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(mi_scores_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(ml_importance_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(pi_scores_df, left_index=True, right_index=True, how='outer')

                tt = pd.DataFrame(rank_model.fit_transform(fi_df), columns=fi_df.columns, index=fi_df.index)
                tt.drop('PI std', axis=1, inplace=True)
                rank = tt.mean(axis=1) / len(tt.columns)

                fi_df['Rank'] = rank
                fi_df.sort_values('Rank', ascending=False, inplace=True)
                fi_df.loc['TT (Sec)'] = [corr_time, cat_corr_time, chi2_time, anova_time, mi_time, ml_time, pi_time, 0, 0]

                if task == 'clf_multiable':
                    fi_df.drop('Corr', axis=1, inplace=True)
            else:
                start_time = time.time()
                corr_df = get_corr_importances(data=data_inside, num_features=num_features+bool_features, target=target)
                corr_df = corr_df.abs()
                corr_time = time.time() - start_time
                start_time = time.time()
                anova_df = get_anova_importances(data=data_inside, features=cat_features+bool_features, target=target)
                anova_time = time.time() - start_time
                start_time = time.time()
                ml_importance_df = get_ml_importances(data=data_inside, num_features=num_features, cat_features=cat_features, bool_features=bool_features, target=target, 
                                        fold_type=fold_type, group=group, nfold=nfold, model_name=ml_model_name, task=task, random_state=random_state, 
                                        early_stopping_rounds=ml_early_stopping_rounds)
                ml_time = time.time() - start_time
                start_time = time.time()
                pi_scores_df = get_permutation_importances(data=data_inside, features=num_features+cat_features+bool_features, target=target, fold_type=fold_type, group=group, nfold=nfold, 
                                                score=score, model_base=model_base, random_state=random_state, n_repeats=pi_n_repeats)
                pi_time = time.time() - start_time

                fi_df = pd.DataFrame()
                fi_df = fi_df.merge(corr_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(anova_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(ml_importance_df, left_index=True, right_index=True, how='outer')
                fi_df = fi_df.merge(pi_scores_df, left_index=True, right_index=True, how='outer')

                tt = pd.DataFrame(rank_model.fit_transform(fi_df), columns=fi_df.columns, index=fi_df.index)
                tt.drop('PI std', axis=1, inplace=True)
                rank = tt.mean(axis=1) / len(tt.columns) 

                fi_df['Rank'] = rank
                fi_df.sort_values('Rank', ascending=False, inplace=True)
                fi_df.loc['TT (Sec)'] = [corr_time, anova_time, ml_time, pi_time, 0, 0]                        
        else:
            print("Unknown method name!")
            print("method names:", ['corr', 'chi2_crosstab', 'chi2', 'anova', 'mi', 'ml', 'pi', 'all'])
            return None

        return fi_df 

    ### Automated Configurations
    # Encoding
    df_enc = data.copy()
    if task in ['clf_multiable', 'clf_binary']:
        df_enc[cat_features] = OrdinalEncoder().fit_transform(df_enc[cat_features])
        df_enc[target] = LabelEncoder().fit_transform(df_enc[[target]])
    else:
        df_enc[cat_features] = OrdinalEncoder().fit_transform(df_enc[cat_features])

    # Base Model Selection For Permutation Importances
    if pi_model_base is None:
        if task in ['clf_multiable', 'clf_binary']:
            pi_model_base = RandomForestClassifier(random_state=random_state)
        else:
            pi_model_base = RandomForestRegressor(random_state=random_state)
    
    # Score Selection For Permutation Importances
    if pi_score is None:
        if task == 'clf_multiable':
            pi_score = 'f1_macro'
        elif task == 'clf_binary':
            pi_score = 'roc_auc'
        elif task == 'reg':
            pi_score = 'neg_root_mean_squared_error'
        else:
            print("Unknown task name!")
            print("task names:", ['clf_binary', 'clf_multiable', 'reg'])
    
    # Reset Index For Fold
    df_enc.reset_index(inplace=True, drop=True)

    ### Feature Importances Calculations
    if type(method) == str:
        fi_df = get_fi_by_name(df_enc, method, pi_model_base, pi_score)
        return fi_df
    elif type(method) == list:
        fi_list = []
        time_list = []
        for m in method:
            start_time = time.time()
            fi_list.append(get_fi_by_name(df_enc, m, pi_model_base, pi_score))
            if m == 'pi':
                time_list.append(time.time() - start_time)
                time_list.append(0)
            else:
                time_list.append(time.time() - start_time)
        fi_df = pd.DataFrame()
        for fi_temp in fi_list:
            fi_df = fi_df.merge(fi_temp, left_index=True, right_index=True, how='outer')
        
        tt = pd.DataFrame(rank_model.fit_transform(fi_df), columns=fi_df.columns, index=fi_df.index)
        if 'PI std' in list(tt.columns):
            tt.drop('PI std', axis=1, inplace=True)
        rank = tt.mean(axis=1) / len(tt.columns) 

        fi_df['Rank'] = rank
        fi_df.sort_values('Rank', ascending=False, inplace=True)
        fi_df.loc['TT (Sec)'] = time_list + [0]
        return fi_df
    else:
        print("Unknown method type!")
        print("method type should be string or list!")


def get_fi_plots(data=None, x=[], y=[], s=200, fontsize=20, alpha=0.8, cmap=cm, ncols=2, figsize=(16, 16)):
    '''
    Plots the feature importances.

    Args:
        data (DataFrame): DataFrame that holds all feature importances values.
        x (str, list): x label features.
        y (str, list): y label features.
        s (int): Size of markers.
        fontsize (int): Size of the font.
        alpha (float): The transparency rate.
        cmap (matplotlib.colors.LinearSegmentedColormap): Color map of the plots.
        ncols (int): Column number of sub plots.
        figsize (tuple): Size of the figure.

    Returns:
        Figure
    '''
    fi_df_droptt = data.drop('TT (Sec)', axis=0)
    if type(x) == list:
        nrows = int(len(x) / ncols)
        if len(x) % ncols != 0:
            nrows += 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], round(nrows*figsize[0]/ncols)))
        for ax, feature_x, feature_y in zip(axes.ravel()[:len(x)], x, y):
            ax.scatter(fi_df_droptt[feature_x], fi_df_droptt[feature_y], s=(fi_df_droptt['Rank']+1)*s, c=fi_df_droptt['Rank'], alpha=alpha, cmap=cmap)
            feature_x_index = np.argwhere(fi_df_droptt.columns==feature_x)
            feature_y_index = np.argwhere(fi_df_droptt.columns==feature_y)
            feature_rank_index = np.argwhere(fi_df_droptt.columns=='Rank')
            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            for i in range(fi_df_droptt.shape[0]):
                ax.annotate(fi_df_droptt.index[i], (fi_df_droptt.iloc[i, feature_x_index.item()], fi_df_droptt.iloc[i, feature_y_index.item()]), 
                            fontsize=(fi_df_droptt.iloc[i, feature_rank_index.item()]+1)*fontsize)
        for ax in axes.ravel()[len(x):]:
            ax.set_visible(False)
    elif type(x) == str:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(fi_df_droptt[x], fi_df_droptt[y], s=(fi_df_droptt['Rank']+1)*s, c=fi_df_droptt['Rank'], alpha=alpha, cmap=cmap)
        feature_x_index = np.argwhere(fi_df_droptt.columns==x)
        feature_y_index = np.argwhere(fi_df_droptt.columns==y)
        feature_rank_index = np.argwhere(fi_df_droptt.columns=='Rank')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        for i in range(fi_df_droptt.shape[0]):
            ax.annotate(fi_df_droptt.index[i], (fi_df_droptt.iloc[i, feature_x_index.item()], fi_df_droptt.iloc[i, feature_y_index.item()]), 
                        fontsize=(fi_df_droptt.iloc[i, feature_rank_index.item()]+1)*fontsize)
    else:
        print("Unknown feature type!")
        print("Both feature type should be string or list!")
    
    fig.suptitle('Feature Importances 2D Scatter Plot', fontsize=20)
    plt.show()
    return fig


def get_corr_importances(data=None, num_features=None, target='target'):
    if len(num_features) < 1:
        return pd.DataFrame(columns=['Corr'])
    corr = data[num_features].corrwith(data[target])
    corr_df = pd.DataFrame(corr, columns=['Corr'], index=num_features)
    corr_df.sort_values('Corr', ascending=False, inplace=True)
    return corr_df


def get_chi2_crosstab_importances(data=None, cat_features=None, target='target'):
    def cramers_corrected_stat(confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
    if len(cat_features) < 1:
        return pd.DataFrame(columns=['Chi_Square_Crosstab'])
        
    cat_corr_dict = {}
        
    for cat_col in cat_features:
            confusion_matrix = pd.crosstab(data[cat_col], data[target]).to_numpy()
            cr = cramers_corrected_stat(confusion_matrix)
            cat_corr_dict[cat_col] = [cr]
            
    cat_corr_df = pd.DataFrame.from_dict(cat_corr_dict).T
    cat_corr_df.columns = ['Chi_Square_Crosstab']
    cat_corr_df.sort_values('Chi_Square_Crosstab', ascending=False, inplace=True)
    return cat_corr_df


def get_chi2_importances(data=None, features=None, target='target'):
    if len(features) < 1:
        return pd.DataFrame(columns=['Chi_Square'])
    chi2_scores = chi2(data[features], data[[target]].values.ravel())
    chi2_df = pd.DataFrame(chi2_scores[0], columns=['Chi_Square'], index=features)
    chi2_df.sort_values('Chi_Square', ascending=False, inplace=True)
    return chi2_df


def get_anova_importances(data=None, features=None, target='target'):
    if len(features) < 1:
        return pd.DataFrame(columns=['ANOVA'])
    anova_scores = f_regression(data[features], data[[target]].values.ravel())
    anova_df = pd.DataFrame(anova_scores[0], columns=['ANOVA'], index=features)
    anova_df.sort_values('ANOVA', ascending=False, inplace=True)
    return anova_df


def get_mutual_info_importances(data=None, features=None, target='target', random_state=0):
    if len(features) < 1:
        return pd.DataFrame(columns=['MI Scores'])
    mi_scores = []
    for feature in features:
        mi_scores.append(mutual_info_classif(data[[feature]], data[target].values.ravel(), random_state=random_state))
    mi_scores_df = pd.DataFrame(mi_scores, columns=["MI Scores"], index=features)
    mi_scores_df = mi_scores_df.sort_values('MI Scores', ascending=False)
    return mi_scores_df


def get_ml_importances(data=None, num_features=[], cat_features=[], bool_features=[], target='target', group='group', fold_type='kf', 
                        nfold=10, model_name='LGBM', task='clf_multiable', random_state=0, early_stopping_rounds=100):
    features = num_features+cat_features+bool_features
    X = data[features]
    y = data[[target]]

    if fold_type == 'kf':
        kf = KFold(n_splits=nfold, random_state=random_state, shuffle=True)
        fold_splits = kf.split(X)
    elif fold_type == 'skf':
        skf = StratifiedKFold(n_splits=nfold, random_state=random_state, shuffle=True)
        fold_splits = skf.split(X, y)
    elif fold_type == 'gkf':
        gkf = GroupKFold(n_splits=nfold)
        fold_splits = gkf.split(X, y, data[[group]])
    else:
        print("Unknown fold_type name!")
        print("fold_type names:", ['kf', 'skf', 'gkf'])
    
    ml_importance_df = pd.DataFrame(np.zeros((len(features))), index=features, columns=[model_name+' Imp.'])

    for train_index, test_index in fold_splits:
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]

        if model_name == 'LGBM':
            num_features = list(num_features) + list(bool_features)
            if task == 'clf_binary':
                class_weight = (len(y_train.values) - np.sum(y_train.values)) / np.sum(y_train.values)
                model = LGBMClassifier(scale_pos_weight=class_weight, random_state=random_state)
            elif task == 'clf_multiable':
                model = LGBMClassifier(class_weight='balanced', random_state=random_state)
            elif task == 'reg':
                model = LGBMRegressor(class_weight='balanced', random_state=random_state)
            else:
                print("Unknown task name!")
                print("task names:", ['clf_binary', 'clf_multiable', 'reg'])
        elif model_name == 'CATBOOST':
            cat_features = list(cat_features) + list(bool_features)
            if task == 'clf_binary':
                class_weight = (len(y_train.values) - np.sum(y_train.values)) / np.sum(y_train.values)
                model = CatBoostClassifier(scale_pos_weight=class_weight, cat_features=cat_features, random_state=random_state)
            elif task == 'clf_multiable':
                target_counts = y[target].value_counts(normalize=True)
                target_counts = target_counts.sort_index()
                class_weights = 1 - target_counts
                model = CatBoostClassifier(loss_function='MultiClass', class_weights=class_weights, cat_features=cat_features, 
                                            random_state=random_state)
            elif task == 'reg':
                model = CatBoostRegressor(cat_features=cat_features, random_state=random_state)
            else:
                print("Unknown task name!")
                print("task names:", ['clf_binary', 'clf_multiable', 'reg'])
            
            for feature in cat_features:
                X_train[feature] = X_train[feature].astype(str)
                X_test[feature] = X_test[feature].astype(str)
        else:
            print("Unknown model name!")
            print("model names:", ['LGBM', 'CATBOOST'])
        
        eval_set = [(X_test, y_test.values.ravel())]
        model.fit(X_train, y_train.values.ravel(), eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=0)

        ml_importance_df[model_name+' Imp.'] += model.feature_importances_ / nfold

        del X_train
        del X_test
        del y_train
        del y_test
        del model
        gc.collect()
    
    ml_importance_df = ml_importance_df.sort_values(model_name+' Imp.', ascending=False)
    return ml_importance_df

def get_permutation_importances(data=None, features=None, target='target', group='group', fold_type='kf', 
                                nfold=10, score='roc_auc', model_base=None, random_state=0, n_repeats=5):
    X = data[features]
    y = data[[target]]

    if fold_type == 'kf':
        kf = KFold(n_splits=nfold, random_state=random_state, shuffle=True)
        fold_splits = kf.split(X)
    elif fold_type == 'skf':
        skf = StratifiedKFold(n_splits=nfold, random_state=random_state, shuffle=True)
        fold_splits = skf.split(X, y)
    elif fold_type == 'gkf':
        gkf = GroupKFold(n_splits=nfold)
        fold_splits = gkf.split(X, y, data[[group]])
    else:
        print("Unknown fold_type name!")
        print("fold_type names:", ['kf', 'skf', 'gkf'])
    
    zeros = np.zeros((len(features), 2))
    permutation_importance_df = pd.DataFrame(zeros, index=features, columns=['PI mean','PI std'])

    for train_index, test_index in fold_splits:
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y.loc[train_index,:], y.loc[test_index,:]
        
        model = clone(model_base)
        model.fit(X_train, y_train.values.ravel())
        r = permutation_importance(model, X_test, y_test.values.ravel(), n_repeats=n_repeats, random_state=random_state, 
                                    scoring=score)
        permutation_importance_df['PI mean'] += r.importances_mean / nfold
        permutation_importance_df['PI std'] += r.importances_std / nfold

        del X_train
        del X_test
        del y_train
        del y_test
        del model
        gc.collect()
    
    permutation_importance_df.sort_values('PI mean', ascending=False, inplace=True)
    return permutation_importance_df


def display_feature_importances(data=None, color_palette='coolwarm', nan_background="white", nan_write_color='black'):
    cm = sns.color_palette(color_palette, as_cmap=True)
    selected_features = list(data.index)
    selected_features.remove('TT (Sec)')
    indices = pd.IndexSlice[selected_features, :]
    display(data.style.background_gradient(cmap=cm, axis=0, subset=indices).applymap(lambda x: f'background: {nan_background}' if pd.isnull(x) else 
                                                                            '').applymap(lambda x: f'color: {nan_write_color}' if pd.isnull(x) else ''))
