import numpy as np
from sklearn import train_test_split

random_state = 42

n = 1024
d = 512



X = np.random.randn(n, d)
y = np.random.randint(2, size=n)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state, stratify=y
)

df_train = X_train
df_test = X_test


df_train.to_csv("X_train.csv", index=False)
df_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)