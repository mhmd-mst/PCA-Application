import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load the csv file data into a pandas dataframe
df = pd.read_csv("COMBO17.csv")
np.random.seed(291385)
num_samples = df.shape[0]
N = 2500

# Sample random training and evaluation sets and write them into csv files
train = np.random.choice(3461, 2500, replace=False)
df_train = df.iloc[train]
mask = df.index.isin(train)
df_test = df[~mask]
df_train.to_csv('COMBO17pca_291385.csv', index_label='Nr')
df_test.to_csv('COMBO17eval_291385.csv', index_label='Nr')


# Drop Nr columns and fill missing values for both datasets
def data_cleaner(df):
    df = df.drop(columns='Nr')
    for i in df.columns:
        if df[i].isna().sum() > 0:
            df[i].fillna(df[i].mean(), inplace=True)
    return df


df_train = data_cleaner(df_train)
df_test = data_cleaner(df_test)

# Split each dataset into variable subset and label subset
X_train = df_train.drop(columns=['Mcz', 'e.Mcz', 'MCzml', 'chi2red'])
y_train = df_train['Mcz']
X_test = df_test.drop(columns=['Mcz', 'e.Mcz', 'MCzml', 'chi2red'])
y_test = df_test['Mcz']

# Define standard scalar and PCA and fit on the training data and transform both sets


znorm = StandardScaler()
pca = PCA(n_components=8)

X_train_norm = znorm.fit_transform(X_train)
X_test_norm = znorm.transform(X_test)

pca.fit(X_train_norm)
Qm = pca.transform(X_train_norm)

# Plot representations for explained variance ratio
print(f'The explained variance ration : {pca.explained_variance_ratio_}')
evr = pca.explained_variance_ratio_
x = np.arange(evr.shape[0])
cum_var = np.cumsum(pca.explained_variance_ratio_)
y = cum_var.tolist()
z = evr.tolist()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(x, z)
axes[1].bar(x, y)
labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

plt.setp(axes, xticks=[0, 1, 2, 3, 4, 5, 6, 7], xticklabels=labels)
# 1st figure styling
plt.sca(axes[0])
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Figure 1', y=-0.3)
# 2nd figure styling
plt.sca(axes[1])
plt.ylabel('Cumulative sum of Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Figure 2', y=-0.3)
plt.tight_layout()
plt.show()

# Bar graph showing the representation of the variables through the first two components
cols=X_train.columns
v=pca.components_
x=np.arange(v[1].shape[0])
locs=np.arange(cols.shape[0])
plt.figure(figsize=(7,12))
plt.barh(x,v[0])
plt.barh(x,v[1])
plt.barh(x,v[2])
plt.legend(('PC 1','PC 2','PC 3'))
plt.yticks(locs,cols)
plt.title('FIGURE 3. The Variables Representation Through the First Two Components')
plt.tight_layout()
plt.show()

# Score Plot Using PC 1
plt.figure()
plt.scatter(Qm[:, 0], np.zeros_like(Qm[:,0]),c=y_train)
plt.grid()
plt.xlabel('scores on PC 1 (41.7 %)')
plt.title('Figure 4. Score Plot Using PC 1')
plt.show()

# Score Plot Using PC 1 and PC 2
cols = X_train.columns
v = pca.components_
v[1].shape[0]
x = np.arange(v[1].shape[0])
locs = np.arange(cols.shape[0])
plt.figure()
plt.scatter(Qm[:, 0], Qm[:, 1],c=y_train)
plt.grid()
plt.xlabel('scores on PC 1 (41.7 %)')
plt.ylabel('scores on PC 2 (16.6 %)')
plt.title('Figure 5. Score Plot Using PC 1 and PC 2')
plt.show()

# Score Plot Using PC 1,2 and 3
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(Qm[:,0],Qm[:,1],Qm[:,2],c=y_train)
plt.grid()
ax.set_xlabel('scores on PC 1 (41.7 %)')
ax.set_ylabel('scores on PC 2 (16.6 %)')
ax.set_zlabel("scores on PC 3 (9.97 %)")
plt.title('Figure 6. Score Plot Using PC 1,2 and 3')
plt.show()

# Apply KNN to predict the redshift
knn = KNeighborsRegressor()
knn.fit(Qm, y_train)
y_pred = knn.predict(pca.transform(X_test_norm))

MAE = sum(abs(y_pred - y_test)) / (num_samples - N)
MRE = sum(abs(y_pred - y_test) / abs(y_test)) / (num_samples - N)
print(f'MAE = {MAE} , MRE = {MRE}')
