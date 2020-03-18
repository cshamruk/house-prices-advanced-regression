COMPETITON = 'house-prices-advanced-regression-techniques'

import subprocess, os

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')

lot_cols = ['LotArea','LotFrontage','MasVnrArea',]
porch_cols = ['WoodDeckSF','OpenPorchSF','ScreenPorch','3SsnPorch','EnclosedPorch',]
bsmt_cols = ['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',]
flr_cols = ['1stFlrSF','2ndFlrSF','GrLivArea',]
num_cols = lot_cols + bsmt_cols + flr_cols + ['GarageArea','LowQualFinSF'] + porch_cols + ['PoolArea','MiscVal',]
cat_cols = list(set(X.columns) - set(num_cols))

print("Train set size:", train.shape)
print("Test set size:", test.shape)
print('START data processing', datetime.now(), )

train_ID = train['Id']
test_ID = test['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Deleting outliers
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train.SalePrice.reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)
print(features.shape)
# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
features['GarageYrBuilt'] = features['GarageYrBuilt'].astype(str)

for col in num_cols:
    if sum(features[col]==0) > 0:
        features['has_'+col] = features[col].apply(lambda x: 1 if x>0 else 0)

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features['Functional'] = features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features[features.has_PoolArea==True]['PoolQC'].fillna('TA')
features[features.has_PoolArea==False]['PoolQC'].fillna('None')

for col in ['GarageYrBlt','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')
features['MSZoning'] = features.groupby(['MSSubClass'])['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

features.update(features[cat_cols].fillna('None'))


features['LotFrontage'] = features.groupby(['Neighborhood','LotConfig'])['LotFrontage'].transform(lambda x: x.fillna(x.mode()[0]))
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)

features['MasVnrType'][features.MasVnrArea>0] = 'BrkFace'
features['MasVnrType'][features.MasVnrArea==0] = 'None'

for col in ('GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)

# Filling in the rest of the NA's

features.update(features[num_cols].fillna(0))

final_features = pd.get_dummies(features).reset_index(drop=True)

X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(X):, :]



submit_call = 'kaggle competitions submit {} -f {} -m {}'.format(COMPETITION, SUBMISSION, MESSAGE)

# subprocess.check_call(submit_call)

num_cols = ['SalePrice','LotArea','LotFrontage','MasVnrArea','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','LowQualFinSF','WoodDeckSF','OpenPorchSF','ScreenPorch','3SsnPorch','EnclosedPorch','PoolArea','MiscVal',]
