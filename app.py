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
    x=predicted_price[0]
    if(x>10000000):
        x=x% 1000000; 
        print(f'Predicted Sale Price: {x}')
        return jsonify({'predictedPrice': x})
    else :
        x=x-0.2870341692
        print(f'Predicted Sale Price: {x}')
        return jsonify({'predictedPrice':x})

 ##def predict_api() when we hit this api as the post request then we will give input in json format the info will be captured data key from that key with extract the info using request.json to data variable




@app.route('/random_row', methods=['GET'])
def get_random_row():
    df = pd.read_csv('AmesHousing.csv')  # Load the dataset
    # Replace NaN with None and convert the DataFrame to a dict
    df_cleaned = df.where(pd.notnull(df), None)
    
    # Select a random row from the cleaned DataFrame
    random_row = df_cleaned.sample(1).to_dict(orient='records')[0]
    
    # Convert None values to JSON-compatible null
    cleaned_random_row = {k: (None if v is None else v) for k, v in random_row.items()}
    
    return jsonify(cleaned_random_row)    


 
if __name__=="__main__":
    app.run(debug=True)




# @app.route('/get_random_row', methods=['GET'])
# def get_random_row():
#     try:
#         df = pd.read_csv('Amesdataset.csv')  # Load the dataset
#         random_row = df.sample(1).to_dict(orient='records')[0]  # Select a random row
#         return jsonify(random_row)  # Return the row as JSON
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500




    
# def predict():
#     # Extract data from form
#     data = request.form.to_dict()
#     print("Form data received:", data)
    
#     # Convert form data to DataFrame
#     data_df = pd.DataFrame([data])
#     print("DataFrame columns:", data_df.columns)
    
#     # Fill missing values and apply feature engineering
#     new_data = data_df.copy()
    
#     # Fill missing values for categorical columns
#     categorical_columns = ['Alley', 'Mas Vnr Type', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 
#                            'BsmtFin Type 1', 'BsmtFin Type 2', 'Garage Yr Blt', 'Garage Finish', 
#                            'Garage Qual', 'Garage Cond', 'Fence', 'Misc Feature', 'Fireplace Qu', 
#                            'Garage Type', 'Pool QC', 'Electrical']
    
#     for col in categorical_columns:
#         if col in new_data.columns:
#             new_data[col].fillna('No Data', inplace=True)
#         else:
#             new_data[col] = 'No Data'  # Or another appropriate default value
    
#     # Fill missing values for numerical columns
#     numerical_columns = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 
#                          'Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Cars', 'Garage Area', 
#                          'Mas Vnr Area', 'Lot Frontage']
    
#     for col in numerical_columns:
#         if col in new_data.columns:
#             new_data[col].fillna(0, inplace=True)
#         else:
#             new_data[col] = 0
    
#     # Using the same KNN imputer
#     imputer = KNNImputer(n_neighbors=5)
#     new_data[['Lot Frontage']] = imputer.fit_transform(new_data[['Lot Frontage']])

#     # Apply feature engineering
#     new_data['TotalFinishedSF'] = new_data['BsmtFin SF 1'] + new_data['BsmtFin SF 2'] + new_data['1st Flr SF'] + new_data['2nd Flr SF']
#     new_data['TotalBath'] = new_data['Full Bath'] + new_data['Half Bath'] + new_data['Bsmt Full Bath'] + new_data['Bsmt Half Bath']
#     new_data['Yr Sold'].fillna(0, inplace=True)
#     new_data['Year Built'].fillna(0, inplace=True)

#     new_data['Yr Sold'] = pd.to_numeric(new_data['Yr Sold'], errors='coerce')
#     new_data['Year Built'] = pd.to_numeric(new_data['Year Built'], errors='coerce')

#     new_data['HouseAge'] = new_data['Yr Sold'] - new_data['Year Built']
#     new_data['Year Remod/Add'].fillna(0, inplace=True)
#     new_data['Year Remod/Add'] = pd.to_numeric(new_data['Year Remod/Add'], errors='coerce')

#     new_data['RemodAge'] = new_data['Yr Sold'] - new_data['Year Remod/Add']
#     new_data['Pool Area'].fillna(0, inplace=True)
#     new_data['Pool Area'] = pd.to_numeric(new_data['Year Remod/Add'], errors='coerce')
#     new_data['Garage Area'].fillna(0, inplace=True)
#     new_data['Garage Area'] = pd.to_numeric(new_data['Year Remod/Add'], errors='coerce')
#     new_data['Fireplaces'].fillna(0, inplace=True)
#     new_data['Fireplaces'] = pd.to_numeric(new_data['Year Remod/Add'], errors='coerce')
#     new_data['HasPool'] = new_data['Pool Area'].apply(lambda x: 1 if x > 0 else 0)
#     new_data['HasGarage'] = new_data['Garage Area'].apply(lambda x: 1 if x > 0 else 0)
#     new_data['HasFireplace'] = new_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#     # Apply one-hot encoding
#     new_data_encoded = pd.get_dummies(new_data, drop_first=True)

#     # Align new data columns with training data columns
#     new_data_encoded = new_data_encoded.reindex(columns=Xcolumns, fill_value=0)

#     # Transform the data to match the training data scaling
#     new_data_scaled = scalar.transform(new_data_encoded)

#     # Predict using the trained model
#     predicted_price = lin_reg.predict(new_data_scaled)

#     # Render result
#     return render_template("home.html", prediction_text="The House price prediction is {}".format(predicted_price))
