
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np

data = pd.read_csv('transport2.csv')


encoder = LabelEncoder()
data["sexe"] = encoder.fit_transform(data["sexe"])


data = data.apply(lambda x: x.fillna(x.median()),axis=0)
X = data.drop(['Public transport', 'num', 'Principal means'], axis=1)
y = data['Public transport']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
numeric_columns = X_train.select_dtypes(exclude='object').columns
categorical_columns = X_train.select_dtypes(include='object').columns

numeric_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='median')),
    ('scaling', MinMaxScaler())
])

categorical_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='most_frequent')),
    ('encoding', OneHotEncoder()),
    ('scaling', MinMaxScaler())
])

processing = ColumnTransformer([
    ('numeric', numeric_features, numeric_columns),
    ('categorical', categorical_features, categorical_columns)
])

def prepare_model(algorithm):
    model = Pipeline(steps= [
        ('processing',processing),
        ('modeling', algorithm)
    ])
    model.fit(X_train, y_train)
    return model


my_model = prepare_model(AdaBoostClassifier())

my_model.score(X_test, y_test)


joblib.dump(my_model, 'model_scaler.pkl')

# input = pd.DataFrame(np.array([45, 0, 3, 1, 1, 1500.0]).reshape(1, -1), columns = ['age', 'sexe', 'sitfam', 'Car', 'Bike', 'revenu'])

def return_preds(model, sample_json):
    input = pd.DataFrame(np.array([sample_json['age'], sample_json['gender'], sample_json['family'], sample_json['car'], sample_json['bike'], sample_json['revenue']]).reshape(1, -1), columns = ['age', 'sexe', 'sitfam', 'Car', 'Bike', 'revenu'])
    prediction = model.predict(input)

    return prediction





