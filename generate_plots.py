import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create plots directory
os.makedirs('data/plots', exist_ok=True)

# Load dataset
df = pd.read_csv('data/Traffic Accident Dataset Process.csv')

print("Generating comprehensive plots for TrafficGuard...")

# PHASE 1: EDA GRAPHS

# 1. Severity Distribution
plt.figure(figsize=(10, 6))
severity_counts = df['Severity'].value_counts().sort_index()
severity_labels = ['Fatal (0)', 'Serious Injury (1)', 'Minor Injury (2)']
plt.bar(severity_labels, severity_counts.values, color=['#e74c3c', '#f39c12', '#27ae60'])
plt.title('Distribution of Accident Severity', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Severity Level', fontsize=12)
for i, v in enumerate(severity_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('data/plots/severity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated severity_distribution.png")

# 2. Univariate Analysis - Numerical Features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0, 0].hist(df['Driver_Age'], bins=30, color='#800000', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribution of Driver Age', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(df['Speed_Limit'], bins=20, color='#c0392b', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribution of Speed Limit', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Speed Limit (km/h)')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(df['Casualties'], bins=20, color='#4a0000', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Casualties', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Casualties')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(df['Time'], bins=30, color='#a93226', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribution of Time (HHMM format)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('data/plots/univariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated univariate_analysis.png")

# 3. Categorical Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
weather_counts = df['Weather'].value_counts().sort_index()
weather_labels = ['Clear', 'Rain', 'Snow', 'Fog']
axes[0, 0].bar(weather_labels, weather_counts.values, color='#800000', alpha=0.8)
axes[0, 0].set_title('Distribution of Weather Conditions', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Count')

road_counts = df['Road_Type'].value_counts().sort_index()
road_labels = ['Highway', 'Intersection', 'Rural']
axes[0, 1].bar(road_labels, road_counts.values, color='#c0392b', alpha=0.8)
axes[0, 1].set_title('Distribution of Road Types', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Count')

vehicle_counts = df['Vehicle_Type'].value_counts().sort_index()
vehicle_labels = ['Motorcycle', 'Car', 'Truck', 'Bus']
axes[1, 0].bar(vehicle_labels, vehicle_counts.values, color='#4a0000', alpha=0.8)
axes[1, 0].set_title('Distribution of Vehicle Types', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Count')

location_counts = df['Location'].value_counts().sort_index()
location_labels = ['Urban', 'Suburban', 'Rural', 'Highway', 'Other']
axes[1, 1].bar(location_labels, location_counts.values, color='#a93226', alpha=0.8)
axes[1, 1].set_title('Distribution of Locations', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data/plots/categorical_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated categorical_distribution.png")

# 4. Pie Charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
severity_pie = df['Severity'].value_counts().sort_index()
severity_pie_labels = ['Fatal (0)', 'Serious Injury (1)', 'Minor Injury (2)']
colors = ['#e74c3c', '#f39c12', '#27ae60']
axes[0, 0].pie(severity_pie.values, labels=severity_pie_labels, autopct='%1.1f%%', 
               colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
axes[0, 0].set_title('Severity Distribution', fontsize=12, fontweight='bold')

weather_pie = df['Weather'].value_counts().sort_index()
weather_pie_labels = ['Clear', 'Rain', 'Snow', 'Fog']
axes[0, 1].pie(weather_pie.values, labels=weather_pie_labels, autopct='%1.1f%%',
               colors=['#3498db', '#9b59b6', '#95a5a6', '#1abc9c'], startangle=90)
axes[0, 1].set_title('Weather Conditions Distribution', fontsize=12, fontweight='bold')

road_pie = df['Road_Type'].value_counts().sort_index()
road_pie_labels = ['Highway', 'Intersection', 'Rural']
axes[1, 0].pie(road_pie.values, labels=road_pie_labels, autopct='%1.1f%%',
               colors=['#e67e22', '#2ecc71', '#f1c40f'], startangle=90)
axes[1, 0].set_title('Road Type Distribution', fontsize=12, fontweight='bold')

vehicle_pie = df['Vehicle_Type'].value_counts().sort_index()
vehicle_pie_labels = ['Motorcycle', 'Car', 'Truck', 'Bus']
axes[1, 1].pie(vehicle_pie.values, labels=vehicle_pie_labels, autopct='%1.1f%%',
               colors=['#9b59b6', '#34495e', '#16a085', '#d35400'], startangle=90)
axes[1, 1].set_title('Vehicle Type Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('data/plots/pie_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated pie_charts.png")

# 5. Violin Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.violinplot(y=df['Driver_Age'], ax=axes[0, 0], color='#800000', alpha=0.7)
axes[0, 0].set_title('Driver Age Distribution (Violin Plot)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Age')

sns.violinplot(y=df['Speed_Limit'], ax=axes[0, 1], color='#c0392b', alpha=0.7)
axes[0, 1].set_title('Speed Limit Distribution (Violin Plot)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Speed Limit (km/h)')

sns.violinplot(y=df['Casualties'], ax=axes[1, 0], color='#4a0000', alpha=0.7)
axes[1, 0].set_title('Casualties Distribution (Violin Plot)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Number of Casualties')

sns.violinplot(y=df['Time'], ax=axes[1, 1], color='#a93226', alpha=0.7)
axes[1, 1].set_title('Time Distribution (Violin Plot)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Time (HHMM)')

plt.tight_layout()
plt.savefig('data/plots/violin_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated violin_plots.png")

# 6. Grouped Violin Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.violinplot(x='Severity', y='Driver_Age', data=df, ax=axes[0, 0], 
               palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 0].set_title('Driver Age Distribution by Severity', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Severity Level')
axes[0, 0].set_ylabel('Driver Age')
axes[0, 0].set_xticklabels(['Fatal', 'Serious', 'Minor'])

sns.violinplot(x='Severity', y='Speed_Limit', data=df, ax=axes[0, 1],
               palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 1].set_title('Speed Limit Distribution by Severity', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Severity Level')
axes[0, 1].set_ylabel('Speed Limit (km/h)')
axes[0, 1].set_xticklabels(['Fatal', 'Serious', 'Minor'])

sns.violinplot(x='Severity', y='Casualties', data=df, ax=axes[1, 0],
               palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 0].set_title('Casualties Distribution by Severity', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Severity Level')
axes[1, 0].set_ylabel('Number of Casualties')
axes[1, 0].set_xticklabels(['Fatal', 'Serious', 'Minor'])

sns.violinplot(x='Severity', y='Time', data=df, ax=axes[1, 1],
               palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 1].set_title('Time Distribution by Severity', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Severity Level')
axes[1, 1].set_ylabel('Time (HHMM)')
axes[1, 1].set_xticklabels(['Fatal', 'Serious', 'Minor'])

plt.tight_layout()
plt.savefig('data/plots/grouped_violin_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated grouped_violin_plots.png")

# 7. Scatter Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0, 0].scatter(df['Driver_Age'], df['Speed_Limit'], alpha=0.5, c='#800000', s=20)
axes[0, 0].set_title('Driver Age vs Speed Limit', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Driver Age')
axes[0, 0].set_ylabel('Speed Limit (km/h)')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(df['Driver_Age'], df['Casualties'], alpha=0.5, c='#c0392b', s=20)
axes[0, 1].set_title('Driver Age vs Casualties', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Driver Age')
axes[0, 1].set_ylabel('Number of Casualties')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(df['Speed_Limit'], df['Casualties'], alpha=0.5, c='#4a0000', s=20)
axes[1, 0].set_title('Speed Limit vs Casualties', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Speed Limit (km/h)')
axes[1, 0].set_ylabel('Number of Casualties')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(df['Time'], df['Casualties'], alpha=0.5, c='#a93226', s=20)
axes[1, 1].set_title('Time vs Casualties', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Time (HHMM)')
axes[1, 1].set_ylabel('Number of Casualties')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/plots/scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated scatter_plots.png")

# 8. Stacked Bar Charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
vehicle_severity_pct = pd.crosstab(df['Vehicle_Type'], df['Severity'], normalize='index') * 100
vehicle_severity_pct.index = vehicle_labels
vehicle_severity_pct.columns = ['Fatal', 'Serious', 'Minor']
vehicle_severity_pct.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                         color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 0].set_title('Severity Distribution by Vehicle Type (Stacked)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Vehicle Type')
axes[0, 0].set_ylabel('Percentage')
axes[0, 0].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].tick_params(axis='x', rotation=0)

weather_severity_pct = pd.crosstab(df['Weather'], df['Severity'], normalize='index') * 100
weather_severity_pct.index = weather_labels
weather_severity_pct.columns = ['Fatal', 'Serious', 'Minor']
weather_severity_pct.plot(kind='bar', stacked=True, ax=axes[0, 1],
                         color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 1].set_title('Severity Distribution by Weather (Stacked)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Weather')
axes[0, 1].set_ylabel('Percentage')
axes[0, 1].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].tick_params(axis='x', rotation=0)

road_severity_pct = pd.crosstab(df['Road_Type'], df['Severity'], normalize='index') * 100
road_severity_pct.index = ['Highway', 'Intersection', 'Rural']
road_severity_pct.columns = ['Fatal', 'Serious', 'Minor']
road_severity_pct.plot(kind='bar', stacked=True, ax=axes[1, 0],
                       color=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 0].set_title('Severity Distribution by Road Type (Stacked)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Road Type')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 0].tick_params(axis='x', rotation=0)

location_severity_pct = pd.crosstab(df['Location'], df['Severity'], normalize='index') * 100
location_severity_pct.index = ['Urban', 'Suburban', 'Rural', 'Highway', 'Other']
location_severity_pct.columns = ['Fatal', 'Serious', 'Minor']
location_severity_pct.plot(kind='bar', stacked=True, ax=axes[1, 1],
                          color=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 1].set_title('Severity Distribution by Location (Stacked)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Location')
axes[1, 1].set_ylabel('Percentage')
axes[1, 1].legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('data/plots/stacked_bar_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated stacked_bar_charts.png")

# 9. Count Plot with Hue
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.countplot(x='Weather', hue='Severity', data=df, ax=axes[0, 0], 
              palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 0].set_title('Weather vs Severity (Count Plot with Hue)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Weather')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend(title='Severity', labels=['Fatal', 'Serious', 'Minor'])
axes[0, 0].set_xticklabels(weather_labels)

sns.countplot(x='Road_Type', hue='Severity', data=df, ax=axes[0, 1],
              palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 1].set_title('Road Type vs Severity (Count Plot with Hue)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Road Type')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(title='Severity', labels=['Fatal', 'Serious', 'Minor'])
axes[0, 1].set_xticklabels(['Highway', 'Intersection', 'Rural'])

sns.countplot(x='Vehicle_Type', hue='Severity', data=df, ax=axes[1, 0],
              palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 0].set_title('Vehicle Type vs Severity (Count Plot with Hue)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Vehicle Type')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(title='Severity', labels=['Fatal', 'Serious', 'Minor'])
axes[1, 0].set_xticklabels(vehicle_labels)

sns.countplot(x='Location', hue='Severity', data=df, ax=axes[1, 1],
              palette=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 1].set_title('Location vs Severity (Count Plot with Hue)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Location')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Severity', labels=['Fatal', 'Serious', 'Minor'])
axes[1, 1].set_xticklabels(['Urban', 'Suburban', 'Rural', 'Highway', 'Other'])

plt.tight_layout()
plt.savefig('data/plots/count_plot_with_hue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated count_plot_with_hue.png")

# 10. Severity by Categorical
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
weather_severity = pd.crosstab(df['Weather'], df['Severity'])
weather_severity.index = weather_labels
weather_severity.columns = ['Fatal', 'Serious', 'Minor']
weather_severity.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 0].set_title('Severity by Weather Condition', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Weather')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend(title='Severity')
axes[0, 0].tick_params(axis='x', rotation=0)

road_severity = pd.crosstab(df['Road_Type'], df['Severity'])
road_severity.index = ['Highway', 'Intersection', 'Rural']
road_severity.columns = ['Fatal', 'Serious', 'Minor']
road_severity.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 1].set_title('Severity by Road Type', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Road Type')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(title='Severity')
axes[0, 1].tick_params(axis='x', rotation=0)

vehicle_severity = pd.crosstab(df['Vehicle_Type'], df['Severity'])
vehicle_severity.index = vehicle_labels
vehicle_severity.columns = ['Fatal', 'Serious', 'Minor']
vehicle_severity.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 0].set_title('Severity by Vehicle Type', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Vehicle Type')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(title='Severity')
axes[1, 0].tick_params(axis='x', rotation=0)

location_severity = pd.crosstab(df['Location'], df['Severity'])
location_severity.index = ['Urban', 'Suburban', 'Rural', 'Highway', 'Other']
location_severity.columns = ['Fatal', 'Serious', 'Minor']
location_severity.plot(kind='bar', ax=axes[1, 1], color=['#e74c3c', '#f39c12', '#27ae60'])
axes[1, 1].set_title('Severity by Location', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Location')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Severity')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('data/plots/severity_by_categorical.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated severity_by_categorical.png")

# 11. Age by Severity Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Severity', y='Driver_Age', data=df, palette=['#e74c3c', '#f39c12', '#27ae60'])
plt.title('Driver Age Distribution by Severity', fontsize=14, fontweight='bold')
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Driver Age', fontsize=12)
plt.xticks([0, 1, 2], ['Fatal', 'Serious', 'Minor'])
plt.tight_layout()
plt.savefig('data/plots/age_by_severity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated age_by_severity_boxplot.png")

# 12. Correlation Matrix
numerical_features = ['Date', 'Time', 'Driver_Age', 'Casualties', 'Speed_Limit', 'Severity']
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated correlation_matrix.png")

# 13. Pairplot
numerical_cols = ['Driver_Age', 'Speed_Limit', 'Casualties', 'Time']
pairplot_df = df[numerical_cols + ['Severity']].copy()
pairplot_df['Severity'] = pairplot_df['Severity'].map({0: 'Fatal', 1: 'Serious', 2: 'Minor'})
sns.pairplot(pairplot_df, vars=numerical_cols, hue='Severity', 
             palette={'Fatal': '#e74c3c', 'Serious': '#f39c12', 'Minor': '#27ae60'},
             diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30}, height=2.5)
plt.suptitle('Pairplot of Numerical Features by Severity', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('data/plots/pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated pairplot.png")

# 14. FacetGrid
g = sns.FacetGrid(df, col='Weather', row='Road_Type', hue='Severity', 
                  palette={0: '#e74c3c', 1: '#f39c12', 2: '#27ae60'},
                  height=3, aspect=1.2, margin_titles=True)
g.map(sns.histplot, 'Driver_Age', bins=15, alpha=0.7)
g.add_legend()
g.fig.suptitle('Driver Age Distribution by Weather and Road Type (colored by Severity)', 
               y=1.02, fontsize=14, fontweight='bold')
plt.savefig('data/plots/facetgrid_weather_road.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated facetgrid_weather_road.png")

# 15. Clustered Heatmap
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
numerical_features = ['Driver_Age', 'Speed_Limit', 'Casualties', 'Time', 'Severity']
correlation_matrix = df[numerical_features].corr()
linkage_matrix = hierarchy.linkage(pdist(correlation_matrix), method='average')
plt.figure(figsize=(10, 8))
sns.clustermap(correlation_matrix, annot=True, cmap='RdYlGn', center=0,
               square=True, linewidths=1, fmt='.2f',
               row_linkage=linkage_matrix, col_linkage=linkage_matrix,
               cbar_kws={"shrink": 0.8})
plt.suptitle('Clustered Correlation Matrix of Numerical Features', 
             y=1.02, fontsize=14, fontweight='bold')
plt.savefig('data/plots/clustered_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated clustered_heatmap.png")

# 16. Parallel Coordinates
from pandas.plotting import parallel_coordinates
parallel_df = df[numerical_features].copy()
parallel_df['Severity_Label'] = parallel_df['Severity'].map({0: 'Fatal', 1: 'Serious', 2: 'Minor'})
plt.figure(figsize=(12, 6))
parallel_coordinates(parallel_df, 'Severity_Label', 
                    color=['#e74c3c', '#f39c12', '#27ae60'],
                    alpha=0.5, linewidth=1)
plt.title('Parallel Coordinates Plot - All Features Colored by Severity', 
          fontsize=14, fontweight='bold')
plt.xlabel('Features')
plt.ylabel('Normalized Values')
plt.grid(alpha=0.3)
plt.legend(title='Severity', loc='upper right')
plt.tight_layout()
plt.savefig('data/plots/parallel_coordinates.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated parallel_coordinates.png")

# 17. Radar Chart
from math import pi
severity_stats = df.groupby('Severity')[numerical_features].mean()
categories = ['Driver_Age', 'Speed_Limit', 'Casualties', 'Time']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
colors = ['#e74c3c', '#f39c12', '#27ae60']
severity_labels_radar = ['Fatal', 'Serious', 'Minor']

for idx, (severity, color) in enumerate(zip([0, 1, 2], colors)):
    values = severity_stats.loc[severity][categories].values.flatten().tolist()
    values += values[:1]
    ax = axes[idx]
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=severity_labels_radar[idx])
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, severity_stats.values.max() * 1.1)
    ax.set_title(f'{severity_labels_radar[idx]} - Average Feature Profile', 
                 fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

plt.tight_layout()
plt.savefig('data/plots/radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated radar_chart.png")

# PHASE 2: FEATURE ENGINEERING

# 18. Engineered features
df['Hour'] = df['Time'] // 100
df['Day_of_Week'] = df['Date'] % 7
df['Part_of_Day'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], 
                          labels=['Night', 'Morning', 'Afternoon', 'Evening'], include_lowest=True)
df['Age_Group'] = pd.cut(df['Driver_Age'], bins=[0, 25, 40, 60, 100],
                        labels=['Young', 'Adult', 'Middle-Aged', 'Senior'], include_lowest=True)

# 19. Line Charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
hour_severity_counts = pd.crosstab(df['Hour'], df['Severity'])
hour_severity_counts.columns = ['Fatal', 'Serious', 'Minor']
hour_severity_counts.plot(ax=axes[0, 0], color=['#e74c3c', '#f39c12', '#27ae60'], linewidth=2)
axes[0, 0].set_title('Severity Trends by Hour of Day', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend(title='Severity')
axes[0, 0].grid(alpha=0.3)

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_severity_counts = pd.crosstab(df['Day_of_Week'], df['Severity'])
day_severity_counts.columns = ['Fatal', 'Serious', 'Minor']
day_severity_counts.index = day_labels
day_severity_counts.plot(ax=axes[0, 1], color=['#e74c3c', '#f39c12', '#27ae60'], linewidth=2, marker='o')
axes[0, 1].set_title('Severity Trends by Day of Week', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(title='Severity')
axes[0, 1].grid(alpha=0.3)

hour_total = df['Hour'].value_counts().sort_index()
axes[1, 0].plot(hour_total.index, hour_total.values, color='#800000', linewidth=2, marker='o')
axes[1, 0].set_title('Total Accidents by Hour of Day', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Total Accidents')
axes[1, 0].grid(alpha=0.3)

day_total = df['Day_of_Week'].value_counts().sort_index()
day_total.index = day_labels
axes[1, 1].plot(day_total.index, day_total.values, color='#c0392b', linewidth=2, marker='o')
axes[1, 1].set_title('Total Accidents by Day of Week', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Total Accidents')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/plots/line_charts_time_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated line_charts_time_trends.png")

# 20. Engineered Features Bar Charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
part_of_day_severity = pd.crosstab(df['Part_of_Day'], df['Severity'])
part_of_day_severity.columns = ['Fatal', 'Serious', 'Minor']
part_of_day_severity.plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 0].set_title('Severity by Part of Day', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Part of Day')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend(title='Severity')
axes[0, 0].tick_params(axis='x', rotation=0)

age_group_severity = pd.crosstab(df['Age_Group'], df['Severity'])
age_group_severity.columns = ['Fatal', 'Serious', 'Minor']
age_group_severity.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 1].set_title('Severity by Age Group', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Age Group')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(title='Severity')
axes[0, 1].tick_params(axis='x', rotation=0)

part_of_day_counts = df['Part_of_Day'].value_counts()
axes[1, 0].bar(part_of_day_counts.index, part_of_day_counts.values, color='#800000', alpha=0.8)
axes[1, 0].set_title('Accident Count by Part of Day', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Part of Day')
axes[1, 0].set_ylabel('Count')

age_group_counts = df['Age_Group'].value_counts()
axes[1, 1].bar(age_group_counts.index, age_group_counts.values, color='#c0392b', alpha=0.8)
axes[1, 1].set_title('Accident Count by Age Group', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Age Group')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data/plots/engineered_features_bar_charts.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated engineered_features_bar_charts.png")

# 21. Hour vs Day Heatmap
hour_day_heatmap = pd.crosstab(df['Hour'], df['Day_of_Week'], values='Severity', aggfunc='count')
hour_day_heatmap.columns = day_labels
plt.figure(figsize=(14, 8))
sns.heatmap(hour_day_heatmap, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Number of Accidents'}, linewidths=0.5)
plt.title('Hour vs Day of Week Heatmap of Accidents', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Hour of Day', fontsize=12)
plt.tight_layout()
plt.savefig('data/plots/hour_day_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated hour_day_heatmap.png")

# 22. KDE Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for severity, color, label in zip([0, 1, 2], ['#e74c3c', '#f39c12', '#27ae60'], 
                                   ['Fatal', 'Serious', 'Minor']):
    sns.kdeplot(df[df['Severity'] == severity]['Time'], ax=axes[0, 0], 
                color=color, label=label, fill=True, alpha=0.3)
axes[0, 0].set_title('Distribution of Time by Severity (KDE)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Time (HHMM)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

for severity, color, label in zip([0, 1, 2], ['#e74c3c', '#f39c12', '#27ae60'],
                                   ['Fatal', 'Serious', 'Minor']):
    sns.kdeplot(df[df['Severity'] == severity]['Driver_Age'], ax=axes[0, 1],
                color=color, label=label, fill=True, alpha=0.3)
axes[0, 1].set_title('Distribution of Driver Age by Severity (KDE)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Driver Age')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

for severity, color, label in zip([0, 1, 2], ['#e74c3c', '#f39c12', '#27ae60'],
                                   ['Fatal', 'Serious', 'Minor']):
    sns.kdeplot(df[df['Severity'] == severity]['Speed_Limit'], ax=axes[1, 0],
                color=color, label=label, fill=True, alpha=0.3)
axes[1, 0].set_title('Distribution of Speed Limit by Severity (KDE)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Speed Limit (km/h)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()

for severity, color, label in zip([0, 1, 2], ['#e74c3c', '#f39c12', '#27ae60'],
                                   ['Fatal', 'Serious', 'Minor']):
    sns.kdeplot(df[df['Severity'] == severity]['Casualties'], ax=axes[1, 1],
                color=color, label=label, fill=True, alpha=0.3)
axes[1, 1].set_title('Distribution of Casualties by Severity (KDE)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Casualties')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('data/plots/kde_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated kde_plots.png")

# 23. Severity by Hour
plt.figure(figsize=(14, 6))
hour_severity = pd.crosstab(df['Hour'], df['Severity'], normalize='index') * 100
hour_severity.columns = ['Fatal', 'Serious', 'Minor']
hour_severity.plot(kind='bar', stacked=True, color=['#e74c3c', '#f39c12', '#27ae60'], figsize=(14, 6))
plt.title('Severity Distribution by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('data/plots/severity_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated severity_by_hour.png")

# 24. Accidents by Hour
plt.figure(figsize=(14, 6))
hour_counts = df['Hour'].value_counts().sort_index()
plt.bar(hour_counts.index, hour_counts.values, color='#800000', alpha=0.8)
plt.title('Accident Count by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Accidents', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('data/plots/accidents_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generated accidents_by_hour.png")

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"Total plots generated: 24")
print(f"Location: data/plots/")
print("="*60)
