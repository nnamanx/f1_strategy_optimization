import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

updated_formula1 = pd.read_csv("updated_cleaned_formula1.csv")
scaler = StandardScaler()


def preprocess_input_data(input_data):    
    features_to_drop = ['date', 'pit_time']
    input_data = input_data.drop(columns=features_to_drop, errors='ignore')
    y = input_data['pit_lap']
    
    # Convert 'pit_duration' to numeric
    input_data['pit_duration'] = pd.to_numeric(input_data['pit_duration'], errors='coerce')
    
    # Drop rows with missing values
    input_data.dropna(inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(input_data.drop(columns=['pit_lap']), y, test_size=0.2, random_state=42)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return (X_train_scaled, X_val_scaled, y_train, y_val)
        
    
def train_model(data):
    X_train, X_val, y_train, y_val = data
    xgb_model = XGBRegressor(n_estimators=50, verbosity=1)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    return xgb_model


def main():
    st.title("Formula 1 Pit Lap Prediction")
    st.sidebar.title("New Data Input")

    # st.subheader('Data Schema')
    # with open("data_schema.txt", "r") as file:
    #     data_schema_text = file.read()
    # st.text(data_schema_text)
    
    # Load the trained XGBoost model
    data = preprocess_input_data(updated_formula1)
    X_train, _, _, _ = data 
    
    # Fit scaler on training data
    scaler.fit(X_train)
    
    # Load the trained XGBoost model
    xgb_model = train_model(data)
    
    # Sidebar - Input fields
    race_id = st.sidebar.number_input("Race ID", value=None, step=1)
    driver_id = st.sidebar.number_input("Driver ID", value=None, step=1)
    circuit_id = st.sidebar.number_input("Circuit ID", value=None, step=1)
    constructor_id = st.sidebar.number_input("Constructor ID", value=None, step=1)
    grid = st.sidebar.number_input("Grid", value=None, step=1)
    position = st.sidebar.number_input("Position", value=None, step=1)
    
    if race_id is not None and any(updated_formula1['raceId'] == race_id):
        if driver_id is not None and any(updated_formula1['driverId'] == driver_id):
            driver_age = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'Driver Age'].iloc[0]
        else:
            driver_age = None

        if circuit_id is not None and any(updated_formula1['circuitId'] == circuit_id):
            laps = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'laps'].iloc[0]
        else:
            laps = None

        if constructor_id is not None and any(updated_formula1['constructorId'] == constructor_id):
            constructor_experience = updated_formula1.loc[updated_formula1['constructorId'] == constructor_id, 'Constructor Experience'].iloc[0]
        else:
            constructor_experience = None

        if grid is not None:
            grid_position = grid
        else:
            grid_position = None

        if position is not None:
            final_position = position
        else:
            final_position = None


        # race_id = st.sidebar.number_input("Race ID", value=None)
         # driver_id = st.sidebar.number_input("Driver ID", value=None)
        driver_age = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'Driver Age'].iloc[0]
    # circuit_id = st.sidebar.number_input("Circuit ID", value=None)
        laps = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'laps'].iloc[0]
    # constructor_id = st.sidebar.number_input("Constructor ID", value=updated_formula1['constructorId'].iloc[0])
    # grid = st.sidebar.number_input("Grid", value=None)
    # position = st.sidebar.number_input("Position", value=None)
        pit_duration = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'pit_duration'].iloc[0]
        seconds = updated_formula1['seconds'].mean()
        fastestLapSpeed = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'fastestLapSpeed'].iloc[0]
        Length = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'Length'].iloc[0]
        Constructor_Experience = updated_formula1.loc[updated_formula1['constructorId'] == constructor_id, 'Constructor Experience'].iloc[0]
        Driver_Experience = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'Driver Experience'].iloc[0]
        Driver_Wins = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'Driver Wins'].iloc[0]
        Constructor_Wins = updated_formula1.loc[updated_formula1['constructorId'] == constructor_id, 'Constructor Wins'].iloc[0]
        Driver_Constructor_Experience = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'Driver Constructor Experience'].iloc[0]
        DNF_Score = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'DNF Score'].iloc[0]
        prev_position = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'prev_position'].iloc[0]
        podium = updated_formula1.loc[updated_formula1['driverId'] == driver_id, 'podium'].iloc[0]
        year = updated_formula1['year'].mean()
        month = updated_formula1['month'].mean()
        day = updated_formula1['day'].mean()
        pit_stop = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'pit_stop'].iloc[0]
        pit_milliseconds = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'pit_milliseconds'].iloc[0]
        Turns = updated_formula1.loc[updated_formula1['circuitId'] == circuit_id, 'Turns'].iloc[0]



        input_data = pd.DataFrame({
            'raceId': [race_id],
            'driverId': [driver_id],
            'Driver Age': [driver_age],
            'laps': [laps],
            'circuitId': [circuit_id],
            'constructorId': [constructor_id],
            'grid': [grid],
            'position': [position],
            'pit_duration': [pit_duration],
            'seconds': [seconds],
            'fastestLapSpeed': [fastestLapSpeed],
            'Length': [Length],
            'Constructor Experience': [Constructor_Experience],
            'Driver Experience': [Driver_Experience],
            'Driver Wins': [Driver_Wins],
            'Constructor Wins': [Constructor_Wins],
            'Driver Constructor Experience': [Driver_Constructor_Experience],
            'DNF Score': [DNF_Score],
            'prev_position': [prev_position],
            'podium': [podium],
            'year': [year],
            'month': [month],
            'day': [day],
            'pit_stop': [pit_stop],
            'pit_milliseconds': [pit_milliseconds],
            'Turns': [Turns]
        })
    

        # Preprocess the input data
        processed_input_data = scaler.transform(input_data) 

        # Make prediction
        prediction = xgb_model.predict(processed_input_data)

        # Display prediction
        st.write("Predicted Pit Lap:", (prediction[0]/2))
    else:
        st.write("Please enter valid Race ID.")

if __name__ == '__main__':
    main()




