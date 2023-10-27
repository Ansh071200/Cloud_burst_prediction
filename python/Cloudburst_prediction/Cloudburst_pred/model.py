import numpy as np
import pandas as pd
import platform

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# importing rfc and metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


os=platform.system()
if os=="Windows":
    # reading the dataset
    df = pd.read_csv("dataset\\Uttarakhand_dataset.csv", encoding='ISO-8859-1')
else:
    # reading the dataset
    df = pd.read_csv("./dataset/Uttarakhand_dataset.csv", encoding='ISO-8859-1')

def rfc(pp,pre,cloud_c,wind_speed,wind_direc,temperature):
    # dropping null values
    df.dropna(inplace = True)
    X = df[['precipitation_probability (%)', 'precipitation (mm)', 'cloudcover_mid (%)', 
        'windspeed_180m (km/h)', 'winddirection_180m ', 'temperature(C)']]
    y = df['CloudBurst']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

    x = np.array([[pp, pre, cloud_c, wind_speed,wind_direc, temperature]])

    # Create and fit the GradientBoostingClassifier
    clf = RandomForestClassifier(random_state=12)
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    # accuracy details
    accuracy = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)

    # print(f'Accuracy: {accuracy}')
    # print(f'Classification Report:\n{classif_report}')
    # print(f'Confusion Matrix: {conf}')


    y_pred = clf.predict(x)
    return y_pred[0].item()

# for testing purpose only 
# print(something(0,97,7.1,195,24.6,0)) 


def Lr(pp, precipitation, cloud_cover, windspeed_, wind_direc, temperature):
    if os=="Windows":
        # Read the dataset
        df = pd.read_csv("dataset\\UttrakhandInt.csv",encoding='ISO-8859-1')
    else:
        # Read the dataset
        df = pd.read_csv("./dataset/UttrakhandInt.csv",encoding='ISO-8859-1')
    X = df[['precipitation_probability (%)', 'precipitation (mm)', 'cloudcover_mid (%)', 
        'windspeed_180m (km/h)', 'winddirection_180m ', 'temperature(C)']]
    y = df[["Intensity %"]]
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Use the trained regression model to predict intensity
    input_data = np.array([pp, precipitation, cloud_cover, windspeed_,wind_direc, temperature]).reshape(1, -1)
    predicted_intensity = model.predict(input_data)
    # Convert the predicted intensity to a float and then format it
    predicted_intensity_float = float(predicted_intensity[0])
    if predicted_intensity_float <= 30:
        # print("Mild Intensity")
        return "Mild Intensity"
    elif 30 < predicted_intensity_float <= 70:
        # print("Moderate Intensity")
        return "Moderate Intensity"
    else:
        # print("Severe Intensity")
        return "Severe Intensity"

# for testing purpose only 
# Lr(97,0.5,36,2.3, 219, 21.3)
