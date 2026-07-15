# 🔴 Crimson Analytics - Traffic Accident Severity Prediction

A comprehensive machine learning system for predicting traffic accident severity with a beautiful Crimson-themed web interface.

## 📊 Project Overview

This project implements a complete data science pipeline for traffic accident severity prediction, including:
- **Data Exploration & Analysis**: Comprehensive EDA with visualizations
- **Machine Learning Models**: Multiple algorithms (Random Forest, XGBoost, SVM, Logistic Regression)
- **Web Application**: Flask backend with responsive Crimson-themed frontend
- **Real-time Predictions**: Interactive UI for instant severity predictions

## 🎯 Features

### Machine Learning Pipeline
- **Data Preprocessing**: Feature engineering, scaling, and encoding
- **Model Training**: 4 different ML algorithms with hyperparameter tuning
- **Model Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Feature Importance**: Analysis of key factors affecting accident severity

### Web Application
- **Splash Screen**: Animated entry with system statistics
- **Dashboard**: Live overview with interactive charts
- **Prediction Tool**: User-friendly form for instant predictions
- **Reports Page**: Comprehensive analytics and visualizations
- **Settings Page**: System configuration and data management

### Design System
- **Crimson Theme**: Professional red and dark grey color palette
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Charts**: Real-time data visualization with Chart.js
- **Modern UI**: Clean, intuitive interface with smooth animations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
cd TrafficGuard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model**
```bash
python train_model.py
```

This will:
- Load and explore the dataset
- Perform feature engineering
- Train multiple ML models
- Select the best model (Random Forest)
- Save the model and preprocessing components
- Generate evaluation plots

4. **Run the web application**
```bash
cd backend
python app.py
```

5. **Open in browser**
Navigate to `http://localhost:5000`

## 📁 Project Structure

```
TrafficGuard/
├── data/
│   ├── Traffic Accident Dataset Process.csv
│   └── plots/                          # EDA visualizations
├── models/
│   ├── traffic_severity_model.pkl     # Trained model
│   ├── preprocessor.pkl               # Preprocessing pipeline
│   └── model_info.pkl                  # Model metadata
├── backend/
│   └── app.py                         # Flask API server
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css              # Crimson theme styles
│   │   └── js/
│   │       └── app.js                # Frontend logic
│   └── templates/
│       └── index.html                # Main application
├── notebooks/
│   └── 01_data_exploration.ipynb     # Jupyter notebook for EDA
├── train_model.py                     # ML training pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```
Returns model status and performance metrics.

### Prediction
```
POST /api/predict
Content-Type: application/json

{
  "location": 0,           # 0-3: Urban, Suburban, Rural, Highway
  "weather": 0,            # 0-3: Clear, Rain, Snow, Fog
  "road_type": 0,          # 0-3: Highway, Intersection, Rural, Urban
  "vehicle_type": 0,       # 0-3: Motorcycle, Car, Truck, Bus
  "driver_age": 35,        # 18-69
  "casualties": 1,          # 0-4
  "speed_limit": 60,       # 40-100
  "time": 1430             # HHMM format
}
```

Returns:
```json
{
  "prediction": "Serious Injury",
  "prediction_code": 1,
  "confidence": 0.87,
  "probabilities": {
    "Fatal": 0.12,
    "Serious Injury": 0.45,
    "Minor Injury": 0.43
  },
  "risk_factors": ["High Speed Zone", "Young Driver"],
  "model_used": "Random Forest",
  "model_accuracy": 0.3404
}
```

### Model Info
```
GET /api/model-info
```
Returns model configuration and performance details.

## 📊 Model Performance

### Best Model: Random Forest
- **Accuracy**: 34.04%
- **F1-Score**: 34.05%
- **Training Parameters**:
  - Max Depth: None
  - Min Samples Split: 5
  - N Estimators: 100

### Feature Importance
1. Minute (time of day)
2. Road Type
3. Weather Conditions
4. Vehicle Type
5. Location
6. Speed Limit
7. Driver Age
8. Casualties

## 🎨 Design System

### Color Palette
- **Primary**: #800000 (Crimson Red)
- **Secondary**: #4a0000 (Dark Crimson)
- **Accent**: #c0392b (Bright Red)
- **Background**: #fdf8f8 (Light Crimson Tint)
- **Text**: #1a1a1a (Dark Grey)
- **Success**: #27ae60 (Green)
- **Warning**: #f39c12 (Orange)
- **Danger**: #e74c3c (Red)

### Severity Classes
- **Fatal (0)**: Red indicator
- **Serious Injury (1)**: Orange indicator
- **Minor Injury (2)**: Green indicator

## 📈 Dataset Information

- **Total Records**: 9,131
- **Features**: 12 (after feature engineering)
- **Target Variable**: Severity (3 classes)
- **Data Balance**: Approximately balanced (33.8%, 33.2%, 33.1%)

### Features
- **Location**: Urban, Suburban, Rural, Highway
- **Weather**: Clear, Rain, Snow, Fog
- **Road Type**: Highway, Intersection, Rural, Urban
- **Vehicle Type**: Motorcycle, Car, Truck, Bus
- **Driver Age**: 18-69 years
- **Casualties**: 0-4
- **Speed Limit**: 40-100 km/h
- **Time**: HHMM format (0-2359)

## 🔬 Methodology

### Phase 1: Data Understanding
- Initial data inspection and statistical analysis
- Distribution analysis of target variable
- Identification of patterns and anomalies

### Phase 2: Data Preprocessing
- Missing value handling
- Feature engineering (Hour, Part_of_Day, Age_Group, Is_Weekend)
- Categorical encoding (One-Hot Encoding)
- Numerical feature scaling (StandardScaler)
- Train-test split (80/20)

### Phase 3: Model Training
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Classifier

### Phase 4: Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis

### Phase 5: Deployment
- Model serialization with joblib
- Flask REST API
- Responsive web interface
- Real-time predictions

## 🛠️ Technologies Used

### Backend
- **Python**: Core programming language
- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with Crimson theme
- **JavaScript (ES6)**: Interactivity
- **Chart.js**: Data visualization

## 📝 Notes

### Model Performance
The current model achieves ~34% accuracy, which is relatively low. This suggests:
- The dataset may have inherent randomness
- Features may not be strongly predictive of severity
- Additional features or data collection could improve performance
- Consider treating this as an ordinal regression problem

### Potential Improvements
1. **Feature Engineering**: Add more sophisticated features
2. **Data Collection**: Gather additional relevant features
3. **Ensemble Methods**: Combine multiple models
4. **Deep Learning**: Try neural network approaches
5. **Ordinal Regression**: Treat severity as ordered categories
6. **Cross-validation**: Use more robust validation strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is for educational and research purposes.

## 👥 Authors

Developed as a comprehensive data science project demonstrating the complete ML pipeline from data exploration to web deployment.

## 🙏 Acknowledgments

- Traffic accident dataset for providing the foundation
- Open source ML libraries (scikit-learn, XGBoost)
- Chart.js for beautiful visualizations
- Flask for the web framework

---

**🔴 Crimson Analytics v2.0.1**
*Traffic Accident Severity Prediction System*
#   T r a f f i c G u a r d  
 