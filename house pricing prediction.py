#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
training=pd.read_csv('train.csv')
training.head()

# %%
from sklearn.preprocessing import OrdinalEncoder
ordinal=OrdinalEncoder()

for col in training:
    if training[col].dtype=='object':
        training[col] = ordinal.fit_transform(training[[col]])

#%%
total=training.isnull().sum().sort_values(ascending=False)
missing_data=pd.DataFrame(total)
missing_data.head(20)

#%%
columns_to_drop = missing_data[missing_data[0] > 81].index
training.drop(columns=columns_to_drop, inplace=True)


training.head()

#%%
for col in training:
    #if training[col].dtype != 'object':  # Check if the column is numeric
        if training[col].isna().sum() > 0:
            training[col].fillna(training[col].mean(), inplace=True)



# %%
variable=training['GrLivArea']
data=training['SalePrice']
plt.scatter(variable,data)

# %%
variable=training["TotalBsmtSF"]
plt.scatter(variable,data)



training.head()

#%%
corr=training.corr()
f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(corr,vmax=.8,square=True)

# %%
k = 10
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(training[cols].values.T)

sns.set(font_scale=1.25)
hm = sns.heatmap(cm, annot=True, square=True, yticklabels=cols, xticklabels=cols)
plt.show()

# %%

def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1


    df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df_no_outliers


training= remove_outliers(training)


# %%
from scipy import stats 

sns.displot(training['SalePrice'])
fig=plt.figure()
res=stats.probplot(training['SalePrice'],plot=plt)

# %%
X=training.drop(['SalePrice'],axis=1)
y=training['SalePrice']


#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# %%
from sklearn.linear_model import LinearRegression

MLR=LinearRegression()

MLR.fit(X,y)
# %%
from sklearn.metrics import r2_score
y_pred=MLR.predict(X_test)
r2=r2_score(y_test,y_pred)

print(r2)
# %%
