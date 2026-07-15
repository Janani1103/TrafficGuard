import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create directories
os.makedirs('data/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*60)
print("TRAFFIC ACCIDENT SEVERITY PREDICTION - ML PIPELINE")
print("="*60)

# ============================================================================
# PHASE 1: Data Understanding & Exploration
# ============================================================================
print("\n[PHASE 1] Loading and exploring data...")

# Load dataset
df = pd.read_csv('data/Traffic Accident Dataset Process.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic info
print("\n--- Data Info ---")
print(df.info())

# Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# Check null values
print("\n--- Null Values ---")
print(df.isnull().sum())

# Severity distribution
print("\n--- Severity Distribution ---")
severity_counts = df['Severity'].value_counts().sort_index()
print(severity_counts)
print("\nSeverity Percentage:")
print(df['Severity'].value_counts(normalize=True).sort_index() * 100)

# ============================================================================
# PHASE 2: Data Preprocessing & Feature Engineering
# ============================================================================
print("\n[PHASE 2] Preprocessing and feature engineering...")

# Step 2.1: Handle missing values (if any)
print("\n--- Handling Missing Values ---")
print(f"Missing values before: {df.isnull().sum().sum()}")
# No missing values based on initial inspection, but we'll add handling for robustness
df = df.fillna({
    'Driver_Age': df['Driver_Age'].median(),
    'Speed_Limit': df['Speed_Limit'].median(),
    'Casualties': df['Casualties'].median(),
    'Weather': df['Weather'].mode()[0],
    'Road_Type': df['Road_Type'].mode()[0],
    'Vehicle_Type': df['Vehicle_Type'].mode()[0],
    'Location': df['Location'].mode()[0]
})
print(f"Missing values after: {df.isnull().sum().sum()}")

# Step 2.2: Feature Engineering
print("\n--- Feature Engineering ---")

# Extract Hour from Time
df['Hour'] = df['Time'] // 100
df['Minute'] = df['Time'] % 100

# Create Part_of_Day feature
def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 21:
        return 2  # Evening
    else:
        return 3  # Night

df['Part_of_Day'] = df['Hour'].apply(get_part_of_day)

# Create Age Groups
def get_age_group(age):
    if age <= 25:
        return 0  # Young
    elif 26 <= age <= 40:
        return 1  # Adult
    elif 41 <= age <= 60:
        return 2  # Middle-aged
    else:
        return 3  # Senior

df['Age_Group'] = df['Driver_Age'].apply(get_age_group)

# Create Is_Weekend from Date (assuming Date is day index)
df['Is_Weekend'] = (df['Date'] % 7 >= 5).astype(int)

print("Features created: Hour, Minute, Part_of_Day, Age_Group, Is_Weekend")

# Step 2.3: Prepare features for modeling
print("\n--- Preparing Features ---")

# Drop columns that won't be used for prediction
columns_to_drop = ['Accident_ID', 'Date', 'Time']
X = df.drop(columns_to_drop + ['Severity'], axis=1)
y = df['Severity']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Define categorical and numerical features
categorical_features = ['Location', 'Weather', 'Road_Type', 'Vehicle_Type', 'Part_of_Day', 'Age_Group', 'Is_Weekend']
numerical_features = ['Driver_Age', 'Casualties', 'Speed_Limit', 'Hour', 'Minute']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 2.4: Train-Test Split
print("\n--- Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# PHASE 3: Model Building & Training
# ============================================================================
print("\n[PHASE 3] Training models...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
    'SVM': SVC(probability=True, random_state=42)
}

# Train and evaluate each model
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model': pipeline
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# ============================================================================
# PHASE 4: Model Evaluation & Tuning
# ============================================================================
print("\n[PHASE 4] Model evaluation and tuning...")

# Display model comparison
print("\n--- Model Comparison ---")
comparison_df = pd.DataFrame(results).T.drop('model', axis=1)
print(comparison_df.sort_values('f1', ascending=False))

# Select best model (based on F1-score)
best_model_name = comparison_df.sort_values('f1', ascending=False).index[0]
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")

# Detailed evaluation of best model
print(f"\n--- Detailed Evaluation: {best_model_name} ---")
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fatal', 'Serious', 'Minor']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
            xticklabels=['Fatal', 'Serious', 'Minor'],
            yticklabels=['Fatal', 'Serious', 'Minor'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('data/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'XGBoost']:
    print("\n--- Feature Importance ---")
    
    # Get feature names after preprocessing
    feature_names = (numerical_features + 
                    list(best_model.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(categorical_features)))
    
    # Get feature importance
    if best_model_name == 'Random Forest':
        importances = best_model.named_steps['classifier'].feature_importances_
    else:  # XGBoost
        importances = best_model.named_steps['classifier'].feature_importances_
    
    # Create DataFrame and sort
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    print(feature_importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='#800000')
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Hyperparameter tuning for best model
print(f"\n--- Hyperparameter Tuning: {best_model_name} ---")

if best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6],
        'classifier__learning_rate': [0.01, 0.1]
    }
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear', 'lbfgs']
    }
else:  # SVM
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear']
    }

grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1-Score: {grid_search.best_score_:.4f}")

# Update best model with tuned parameters
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted')

print(f"\nTuned Model Performance:")
print(f"  Accuracy: {tuned_accuracy:.4f}")
print(f"  F1-Score: {tuned_f1:.4f}")

# ============================================================================
# PHASE 5: Save Model and Preprocessing Components
# ============================================================================
print("\n[PHASE 5] Saving model and preprocessing components...")

# Save the best model
joblib.dump(best_model, 'models/traffic_severity_model.pkl')
print("Model saved: models/traffic_severity_model.pkl")

# Save feature names and preprocessing info
model_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'feature_names': numerical_features + list(best_model.named_steps['preprocessor']
                                                .named_transformers_['cat']
                                                .get_feature_names_out(categorical_features)),
    'best_model_name': best_model_name,
    'accuracy': tuned_accuracy,
    'f1_score': tuned_f1
}

joblib.dump(model_info, 'models/model_info.pkl')
print("Model info saved: models/model_info.pkl")

# Save the preprocessor separately for use in API
preprocessor_only = best_model.named_steps['preprocessor']
joblib.dump(preprocessor_only, 'models/preprocessor.pkl')
print("Preprocessor saved: models/preprocessor.pkl")

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {tuned_accuracy:.4f}")
print(f"F1-Score: {tuned_f1:.4f}")
print("\nModel and preprocessing components saved to 'models/' directory")
print("Plots saved to 'data/plots/' directory")
