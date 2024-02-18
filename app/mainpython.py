import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
# Import the dataset
df = pd.read_csv('data.csv')
# Drop the ID number
df = df.drop(['id'], axis=1)

# Drop the Unnamed: 32 column
df = df.drop(['Unnamed: 32'], axis=1)
print(df.info())
# Encode the diagnosis variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# Normalize the data
scaler = StandardScaler()
scaler.fit(df.drop('diagnosis', axis=1))
scaled_features = scaler.transform(df.drop('diagnosis', axis=1))

# Create the dataframe
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Create the X and y variables
X = df_feat
y = df['diagnosis']

# Create the model
logmodel = LogisticRegression()
logmodel.fit(X, y)
#Test the model
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# Test the model
predictions = logmodel.predict(X_test)
# Print the report
print(classification_report(y_test, predictions))

# Predict the diagnosis of a tumor
logmodel.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])
# Save the model
pickle.dump(logmodel, open('model.pkl', 'wb'))
# Save the scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))