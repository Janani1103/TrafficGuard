from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'traffic_severity_model.pkl')
MODEL_INFO_PATH = os.path.join(BASE_DIR, 'models', 'model_info.pkl')

print("Loading model and preprocessing components...")
model = joblib.load(MODEL_PATH)
model_info = joblib.load(MODEL_INFO_PATH)
print("Model loaded successfully!")

def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 0 
    elif 12 <= hour < 17:
        return 1 
    elif 17 <= hour < 21:
        return 2
    else:
        return 3 

def get_age_group(age):
    if age <= 25:
        return 0 
    elif 26 <= age <= 40:
        return 1  
    elif 41 <= age <= 60:
        return 2
    else:
        return 3 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': model_info['best_model_name'],
        'accuracy': model_info['accuracy'],
        'f1_score': model_info['f1_score']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict accident severity from input features
    
    Expected JSON input:
    {
        "location": int (0-4),
        "weather": int (0-3),
        "road_type": int (0-3),
        "vehicle_type": int (0-3),
        "driver_age": int (18-69),
        "casualties": int (0-4),
        "speed_limit": int (40-100),
        "time": int (0-2359)
    }
    """
    try:
        data = request.get_json()
       
        location = data.get('location')
        weather = data.get('weather')
        road_type = data.get('road_type')
        vehicle_type = data.get('vehicle_type')
        driver_age = data.get('driver_age')
        casualties = data.get('casualties')
        speed_limit = data.get('speed_limit')
        time = data.get('time')
       
        if None in [location, weather, road_type, vehicle_type, 
                   driver_age, casualties, speed_limit, time]:
            return jsonify({
                'error': 'Missing required fields'
            }), 400
       
        hour = time // 100
        minute = time % 100
        part_of_day = get_part_of_day(hour)
        age_group = get_age_group(driver_age)
        is_weekend = 0 
       
        features = pd.DataFrame({
            'Location': [location],
            'Weather': [weather],
            'Road_Type': [road_type],
            'Vehicle_Type': [vehicle_type],
            'Driver_Age': [driver_age],
            'Casualties': [casualties],
            'Speed_Limit': [speed_limit],
            'Hour': [hour],
            'Minute': [minute],
            'Part_of_Day': [part_of_day],
            'Age_Group': [age_group],
            'Is_Weekend': [is_weekend]
        })
        
        # Predict using the complete pipeline (includes preprocessing)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map prediction to severity label
        severity_labels = {
            0: 'Fatal',
            1: 'Serious Injury',
            2: 'Minor Injury'
        }
        
        # Get confidence (max probability)
        confidence = float(max(probabilities))
        
        # Create response with probabilities for each class
        class_probabilities = {
            'Fatal': float(probabilities[0]),
            'Serious Injury': float(probabilities[1]),
            'Minor Injury': float(probabilities[2])
        }
        
        # Generate risk factors based on input
        risk_factors = []
        if speed_limit >= 80:
            risk_factors.append('High Speed Zone')
        if driver_age <= 25:
            risk_factors.append('Young Driver')
        elif driver_age >= 60:
            risk_factors.append('Senior Driver')
        if weather in [1, 2, 3]:  # Rain, Snow, Fog
            weather_conditions = ['Rain', 'Snow', 'Fog']
            risk_factors.append(f'Adverse Weather ({weather_conditions[weather-1]})')
        if location == 3:  # Highway
            risk_factors.append('Highway Location')
        if casualties >= 2:
            risk_factors.append('Multiple Casualties')
        
        response = {
            'prediction': severity_labels[prediction],
            'prediction_code': int(prediction),
            'confidence': confidence,
            'probabilities': class_probabilities,
            'risk_factors': risk_factors,
            'model_used': model_info['best_model_name'],
            'model_accuracy': model_info['accuracy']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/model-info')
def get_model_info():
    """Return model information"""
    return jsonify({
        'best_model_name': model_info['best_model_name'],
        'accuracy': model_info['accuracy'],
        'f1_score': model_info['f1_score'],
        'categorical_features': model_info['categorical_features'],
        'numerical_features': model_info['numerical_features']
    })

@app.route('/api/filter', methods=['POST'])
def filter_data():
    """
    Filter accident data based on provided criteria
    
    Expected JSON input:
    {
        "location": int or "all",
        "weather": int or "all",
        "road_type": int or "all",
        "vehicle_type": int or "all"
    }
    """
    try:
        data = request.get_json()
        
        # Load the dataset
        df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Traffic Accident Dataset Process.csv'))
        
        # Apply filters
        if data.get('location') and data.get('location') != 'all':
            df = df[df['Location'] == int(data['location'])]
        
        if data.get('weather') and data.get('weather') != 'all':
            df = df[df['Weather'] == int(data['weather'])]
        
        if data.get('road_type') and data.get('road_type') != 'all':
            df = df[df['Road_Type'] == int(data['road_type'])]
        
        if data.get('vehicle_type') and data.get('vehicle_type') != 'all':
            df = df[df['Vehicle_Type'] == int(data['vehicle_type'])]
        
        # Calculate statistics
        severity_counts = df['Severity'].value_counts().to_dict()
        total_records = int(len(df))
        
        # Calculate severity by weather
        weather_severity = df.groupby('Weather')['Severity'].value_counts().unstack(fill_value=0)
        weather_data = {
            'labels': ['Clear', 'Rain', 'Snow', 'Fog'],
            'fatal': [int(weather_severity.get(0, {}).get(0, 0)), int(weather_severity.get(1, {}).get(0, 0)), 
                      int(weather_severity.get(2, {}).get(0, 0)), int(weather_severity.get(3, {}).get(0, 0))],
            'serious': [int(weather_severity.get(0, {}).get(1, 0)), int(weather_severity.get(1, {}).get(1, 0)),
                        int(weather_severity.get(2, {}).get(1, 0)), int(weather_severity.get(3, {}).get(1, 0))],
            'minor': [int(weather_severity.get(0, {}).get(2, 0)), int(weather_severity.get(1, {}).get(2, 0)),
                      int(weather_severity.get(2, {}).get(2, 0)), int(weather_severity.get(3, {}).get(2, 0))]
        }
        
        # Calculate severity by road type
        road_severity = df.groupby('Road_Type')['Severity'].value_counts().unstack(fill_value=0)
        road_data = {
            'labels': ['Highway', 'Intersection', 'Rural', 'Urban'],
            'fatal': [int(road_severity.get(0, {}).get(0, 0)), int(road_severity.get(1, {}).get(0, 0)),
                      int(road_severity.get(2, {}).get(0, 0)), int(road_severity.get(3, {}).get(0, 0))],
            'serious': [int(road_severity.get(0, {}).get(1, 0)), int(road_severity.get(1, {}).get(1, 0)),
                        int(road_severity.get(2, {}).get(1, 0)), int(road_severity.get(3, {}).get(1, 0))],
            'minor': [int(road_severity.get(0, {}).get(2, 0)), int(road_severity.get(1, {}).get(2, 0)),
                      int(road_severity.get(2, {}).get(2, 0)), int(road_severity.get(3, {}).get(2, 0))]
        }
        
        response = {
            'total_records': total_records,
            'severity_counts': {
                'fatal': int(severity_counts.get(0, 0)),
                'serious': int(severity_counts.get(1, 0)),
                'minor': int(severity_counts.get(2, 0))
            },
            'weather_data': weather_data,
            'road_data': road_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/plots/<filename>')
def serve_plot(filename):
    """Serve plot images from the data/plots directory"""
    try:
        return send_from_directory(os.path.join(BASE_DIR, 'data', 'plots'), filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/plots')
def list_plots():
    """List all available plot files"""
    try:
        plots_dir = os.path.join(BASE_DIR, 'data', 'plots')
        if os.path.exists(plots_dir):
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.html'))]
            return jsonify({'plots': sorted(plot_files)})
        else:
            return jsonify({'plots': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
