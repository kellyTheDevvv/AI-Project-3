import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score

#Load the dataset with pandas
df = pd.read_csv('Iris.csv')

#Drop irrelevant colums
df = df.drop('Id', axis=1)

#Handle missing values
df = df.dropna() #Drop rows with missing values if any

#Encode the label 'Species' to numerical values
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

#Separate features(x) and target(y) variables
x = df.drop('Species', axis=1)
y = df['Species']

#Split the data into training and testing sets(80% for training, 20% for testing)
x_train,y_train,x_test,y_test = train_test_split(x, y, test_size=0.2, random_size=42)

#Initiate the DecisionTreeClassifier model
model = DecisionTreeClassifier(random_size=42)
model.fit(x_train, y_train)

#Make predictions on the test set
y_pred = model.predict(x_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  #How often is the classifier correct?
precision = precision_score(y_test, y_pred, average='macro') #Precision averaged across classes
recall = recall_score(y_test, y_pred, average='macro') #Recall averaged across classes

print(f'Accuracy :{accuracy}')
print(f'Precision :{precision}')
print(f'Recall :{recall}')