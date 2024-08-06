
import pickle
from sklearn.impute import KNNImputer
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__) # basic flask app created __name__ is starting point of the application from where it will run
## Load the model
lin_reg=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
Xcolumns=pickle.load(open('col_name.pkl','rb'))
## when we hit the url we will be directed here(Home Page)
@app.route('/')
def home():
    return render_template('home.html')
# We will create an predict API so using postman or any other tool we will jsut send the request to our app we will get output.Also using methods as post bcz it will be an post request bcz from my side i will provide some input , That will actually capture the input, that input we will give to our model, to predict the output  
@app.route('/predict_api', methods=['POST'])
def predict_api():
    new_data = request.json['data']
    new_data_df = pd.DataFrame([new_data])
    
    # Fill missing values and apply feature engineering
    # Fill missing values for categorical columns
    fill_values_categorical = {
        'Alley': 'No Alley',
        'Mas Vnr Type': 'None',
        'Bsmt Qual': 'No Basement',
        'Bsmt Cond': 'No Basement',
        'Bsmt Exposure': 'No Exposure',
        'BsmtFin Type 1': 'No Basement',
        'BsmtFin Type 2': 'No Basement',
        'Garage Yr Blt': 0,
        'Garage Finish': 'No Garage',
        'Garage Qual': 'No Garage',
        'Garage Cond': 'No Garage',
        'Fence': 'No Fence',
        'Misc Feature': 'None',
        'Fireplace Qu': 'No Fireplace',
        'Garage Type': 'No Garage',
        'Pool QC': 'No Pool',
        'Electrical': 'SBrkr'
    }
    
    new_data_df.fillna(value=fill_values_categorical, inplace=True)

    # Fill missing values for numerical columns
    fill_values_numerical = {
        'BsmtFin SF 1': 0,
        'BsmtFin SF 2': 0,
        'Bsmt Unf SF': 0,
        'Total Bsmt SF': 0,
        'Bsmt Full Bath': 0,
        'Bsmt Half Bath': 0,
        'Garage Cars': 0,
        'Garage Area': 0,
        'Mas Vnr Area': 0.0
    }
    
    new_data_df.fillna(value=fill_values_numerical, inplace=True)

    # Using the same KNN imputer for Lot Frontage
    imputer = KNNImputer(n_neighbors=5)
    new_data_df[['Lot Frontage']] = imputer.fit_transform(new_data_df[['Lot Frontage']])

    # Apply feature engineering
    new_data_df['TotalFinishedSF'] = new_data_df['BsmtFin SF 1'] + new_data_df['BsmtFin SF 2'] + new_data_df['1st Flr SF'] + new_data_df['2nd Flr SF']
    new_data_df['TotalBath'] = new_data_df['Full Bath'] + new_data_df['Half Bath'] + new_data_df['Bsmt Full Bath'] + new_data_df['Bsmt Half Bath']
    new_data_df['HouseAge'] = new_data_df['Yr Sold'] - new_data_df['Year Built']
    new_data_df['RemodAge'] = new_data_df['Yr Sold'] - new_data_df['Year Remod/Add']
    new_data_df['HasPool'] = new_data_df['Pool Area'].apply(lambda x: 1 if x > 0 else 0)
    new_data_df['HasGarage'] = new_data_df['Garage Area'].apply(lambda x: 1 if x > 0 else 0)
    new_data_df['HasFireplace'] = new_data_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    # Encode categorical variables
    new_data_encoded = pd.get_dummies(new_data_df, drop_first=True)

    # Align new data columns with top features (Xcolumns)
    new_data_encoded = new_data_encoded.reindex(columns=Xcolumns, fill_value=0)

    # Transform the data to match the training data scaling
    new_data_scaled = scalar.transform(new_data_encoded)

    # Predict using the trained model
    predicted_price = lin_reg.predict(new_data_scaled)

    print(f'Predicted Sale Price: {predicted_price[0]}')
    return jsonify(predicted_price[0])

 ##def predict_api() when we hit this api as the post request then we will give input in json format the info will be captured data key from that key with extract the info using request.json to data variable

 
if __name__=="__main__":
    app.run(debug=True)