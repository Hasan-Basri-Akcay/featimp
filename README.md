# Don't Miss The Contents
Medium: https://medium.com/subscribe/@hasan.basri.akcay
LinkedIn: https://www.linkedin.com/in/hasan-basri-akcay

# featimp
Feature importance for machine learning. Helps with feature understanding, calculating feature importances, feature debugging, and leakage detection.

## Installation
```
pip install featimp
```

## Using featimp
Detailed [Medium post](https://medium.com/@hasan.basri.akcay) on using featimp.

There are a lot of feature importance techniques and each technique calculates different importance. Some of them are suitable for numerical to numerical importance, some of them are ideal for categorical to numerical significance and some of them are suitable for categorical to categorical importance. featimp automatically calculates feature importances and ranks them for you.

```
from featimp import get_feature_importances

fi_df = get_feature_importances(data=df_diabetes, num_features=num_features, cat_features=cat_features, 
                                target='target_clf', task='clf_multiable', method='all')
fi_df.style.background_gradient(cmap=cm)
```
<img src="/outputs/fi_df.png?raw=true"/>

```
from featimp import get_fi_plots

_ = get_fi_plots(data=fi_df, x=['LGBM Imp.', 'LGBM Imp.'], y=['PI mean', 'ANOVA'])
```
<img src="/outputs/feature_importances_2d.png?raw=true"/>
<img src="/outputs/feature_importance_3d.gif?raw=true"/>

## Feature Importances Schema
<img src="/outputs/Feature Importances Techniques.png?raw=true"/>
