# ---------------------------------------
# COMPLETE FLORIDA FIRE RISK VISUALIZATION WITH NEURAL NETWORK AND GOES-16 DATA
# ---------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print("Starting Florida Fire Risk Visualization...")

# ---------------------------------------
# STEP 1: Load data and create grid
# ---------------------------------------
# Define Florida boundaries
MIN_LON, MAX_LON = -87.6, -80.0
MIN_LAT, MAX_LAT = 24.5, 31.0
GRID_SIZE = 0.1  # grid cell size in degrees

# Load fire data
fl2024_path = r"C:\Users\jason_vxe1\Downloads\FireNeuralNetwork\fl2024_firedata.csv"
wfigs_path = r"C:\Users\jason_vxe1\Downloads\FireNeuralNetwork\WFIGS_Incident_Locations_YearToDate_3640042155344676898.csv"

print("Loading fire datasets...")
fl2024 = pd.read_csv(fl2024_path)
fl2024 = fl2024[['f_lat', 'f_lon', 'f_area_acr', 'dttime_est', 'county']].rename(columns={
    'f_lat': 'latitude', 'f_lon': 'longitude', 'f_area_acr': 'area_acres', 'dttime_est': 'datetime_est'
})
fl2024[['latitude', 'longitude', 'area_acres']] = fl2024[['latitude', 'longitude', 'area_acres']].apply(pd.to_numeric,
                                                                                                        errors='coerce')
fl2024.dropna(subset=['latitude', 'longitude'], inplace=True)

wfigs = pd.read_csv(wfigs_path)
wfigs = wfigs[wfigs['POOState'] == 'FL']
wfigs = wfigs[['InitialLatitude', 'InitialLongitude', 'IncidentName', 'POOCounty', 'FireCause']]
wfigs.dropna(subset=['InitialLatitude', 'InitialLongitude'], inplace=True)

# Load GOES-16 precipitation data (from shortcode)
goes_path = r"C:\Users\jason_vxe1\Downloads\FireNeuralNetwork\GOES-16 Data"
goes_files = glob.glob(os.path.join(goes_path, "*.csv"))

has_goes_data = False
if goes_files:
    print(f"Loading {len(goes_files)} GOES-16 data files...")
    goes_data = []
    for file in goes_files:
        df = pd.read_csv(file)
        goes_data.append(df)
    if goes_data:
        goes_combined = pd.concat(goes_data, ignore_index=True)
        print(f"GOES-16 data loaded: {len(goes_combined)} data points")
        goes_gdf = gpd.GeoDataFrame(
            goes_combined,
            geometry=gpd.points_from_xy(goes_combined.longitude, goes_combined.latitude),
            crs="EPSG:4326"
        )
        has_goes_data = True
else:
    print("No GOES-16 data files found. Will use synthetic precipitation data.")

# Try to get Florida state boundary
try:
    # Try multiple sources for Florida boundary
    print("Attempting to load Florida boundary...")

    # Option 1: Direct URL to a GeoJSON with Florida boundary
    try:
        florida = gpd.read_file(
            'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
        florida = florida[florida['name'] == 'Florida']
        if not florida.empty:
            print("Florida boundary loaded successfully from GitHub.")
            has_florida_shape = True
        else:
            raise ValueError("Florida not found in dataset")
    except:
        # Option 2: Create rough Florida boundary from grid to showcase visualization
        print("Using simplified Florida boundary...")
        x_coords = np.array([-87.6, -86.5, -85.0, -84.0, -83.0, -82.0, -82.0, -81.0, -80.5, -80.0, -80.5, -81.3, -82.2,
                             -83.1, -84.0, -85.0, -86.0, -87.0, -87.6])
        y_coords = np.array([31.0, 31.0, 31.0, 31.0, 30.7, 30.4, 29.8, 29.1, 28.4, 26.8, 25.2, 24.5, 24.8,
                             25.4, 26.0, 26.5, 27.8, 29.0, 30.5])
        florida_poly = Polygon(zip(x_coords, y_coords))
        florida = gpd.GeoDataFrame(geometry=[florida_poly], crs="EPSG:4326")
        has_florida_shape = True

except Exception as e:
    print(f"Could not create Florida boundary: {e}")
    has_florida_shape = False

# Create Florida grid
print("Creating grid over Florida...")
grid_cells = []
for lon in np.arange(MIN_LON, MAX_LON, GRID_SIZE):
    for lat in np.arange(MIN_LAT, MAX_LAT, GRID_SIZE):
        grid_cells.append(box(lon, lat, lon + GRID_SIZE, lat + GRID_SIZE))

grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
grid['cell_id'] = grid.index

# If we have Florida boundary, clip grid to state boundary
if has_florida_shape:
    print("Clipping grid to Florida boundary...")
    grid = gpd.clip(grid, florida)

# ---------------------------------------
# STEP 2: Map fires to grid cells
# ---------------------------------------
print("Mapping fires to grid cells...")
# Convert fire locations to GeoDataFrames
fl_gdf = gpd.GeoDataFrame(
    fl2024,
    geometry=gpd.points_from_xy(fl2024.longitude, fl2024.latitude),
    crs="EPSG:4326"
)
w_gdf = gpd.GeoDataFrame(
    wfigs,
    geometry=gpd.points_from_xy(wfigs.InitialLongitude, wfigs.InitialLatitude),
    crs="EPSG:4326"
)

# Join fires to grid cells
fl_join = gpd.sjoin(fl_gdf, grid, how='left', predicate='within')
w_join = gpd.sjoin(w_gdf, grid, how='left', predicate='within')

# Count fires per cell
fl_counts = fl_join.groupby('cell_id').size().reset_index(name='fl_fire_count')
w_counts = w_join.groupby('cell_id').size().reset_index(name='wfigs_fire_count')

# Merge fire counts back to grid
grid = grid.merge(fl_counts, on='cell_id', how='left')
grid = grid.merge(w_counts, on='cell_id', how='left')

# Fill NAs and calculate fire presence
grid['fl_fire_count'] = grid['fl_fire_count'].fillna(0).astype(int)
grid['wfigs_fire_count'] = grid['wfigs_fire_count'].fillna(0).astype(int)
grid['total_fires'] = grid['fl_fire_count'] + grid['wfigs_fire_count']
grid['has_fire'] = (grid['total_fires'] > 0).astype(int)

# ---------------------------------------
# STEP 3: Process GOES-16 data and risk factors
# ---------------------------------------
print("Processing fire risk factors...")

# Process GOES-16 data if available (from shortcode)
if has_goes_data:
    print("Processing GOES-16 precipitation data...")
    goes_join = gpd.sjoin(goes_gdf, grid, how='left', predicate='within')

    precip_stats = goes_join.groupby('cell_id')['precip'].agg(['mean', 'max', 'min', 'std', 'count']).reset_index()
    precip_stats.columns = ['cell_id', 'precip_mean', 'precip_max', 'precip_min', 'precip_std', 'precip_count']

    grid = grid.merge(precip_stats, on='cell_id', how='left')
    grid[['precip_mean', 'precip_max', 'precip_min', 'precip_std', 'precip_count']] = grid[
        ['precip_mean', 'precip_max', 'precip_min', 'precip_std', 'precip_count']
    ].fillna(0)

    grid['dry_area'] = (grid['precip_mean'] < 0.1).astype(int)
    grid['precip_variability'] = grid['precip_std'] / (grid['precip_mean'] + 0.1)
    grid['precipitation'] = grid['precip_mean']

    print(f"Precipitation data mapped to {len(precip_stats)} grid cells")
else:
    # Use synthetic precipitation data as in original longer code
    print("Using synthetic precipitation data...")
    grid['precipitation'] = np.random.gamma(2, 2, len(grid))
    for idx, row in grid[grid['has_fire'] == 1].iterrows():
        grid.loc[idx, 'precipitation'] = max(0.1, grid.loc[idx, 'precipitation'] * np.random.uniform(0.3, 0.7))

    # Create synthetic precipitation-related features to match GOES data structure
    grid['precip_mean'] = grid['precipitation']
    grid['precip_max'] = grid['precipitation'] * np.random.uniform(1.2, 2.0, len(grid))
    grid['precip_min'] = grid['precipitation'] * np.random.uniform(0.1, 0.8, len(grid))
    grid['precip_std'] = grid['precipitation'] * np.random.uniform(0.1, 0.5, len(grid))
    grid['precip_count'] = np.random.poisson(10, len(grid))
    grid['dry_area'] = (grid['precip_mean'] < 0.1).astype(int)
    grid['precip_variability'] = grid['precip_std'] / (grid['precip_mean'] + 0.1)

# Generate additional risk factors (from original longer code)
np.random.seed(42)  # For reproducibility

# Generate lightning strikes - more likely near actual fires
grid['lightning_strikes'] = np.random.poisson(0.5, len(grid))
for idx, row in grid[grid['has_fire'] == 1].iterrows():
    grid.loc[idx, 'lightning_strikes'] += np.random.randint(1, 5)

# Temperature patterns - higher in areas with fires
grid['temperature'] = 28 + np.random.normal(0, 2, len(grid))
for idx, row in grid[grid['has_fire'] == 1].iterrows():
    grid.loc[idx, 'temperature'] += np.random.uniform(1, 4)

# Vegetation/fuel - higher in areas with fires
grid['vegetation_density'] = np.random.beta(2, 2, len(grid))
for idx, row in grid[grid['has_fire'] == 1].iterrows():
    grid.loc[idx, 'vegetation_density'] = min(1.0, grid.loc[idx, 'vegetation_density'] + np.random.uniform(0.1, 0.3))

# ---------------------------------------
# STEP 4: Train Neural Network Model
# ---------------------------------------
print("Training neural network model...")

# Prepare features and target - now including GOES-16 data features when available
features = ['lightning_strikes', 'temperature', 'vegetation_density', 'precipitation']
if has_goes_data:
    features += ['precip_max', 'precip_variability', 'dry_area']  # Add GOES-16 specific features

X = grid[features].values
y = grid['has_fire'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Fire occurrence in training: {sum(y_train)} out of {len(y_train)} ({sum(y_train) / len(y_train):.2%})")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define class weights to address imbalance
fire_ratio = sum(y_train) / len(y_train)
class_weight = {
    0: 1 / 2 * (1 - fire_ratio),
    1: 1 / 2 * fire_ratio
}
print(f"Using class weights: {class_weight}")

# Build neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# Evaluate model
test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {test_results[1]:.4f}, AUC: {test_results[2]:.4f}")

# Make predictions for all grid cells
grid_features_scaled = scaler.transform(grid[features].values)
grid['predicted_risk'] = model.predict(grid_features_scaled).flatten()

# Normalize predicted risk to 0-100 scale
grid['fire_risk_pct'] = 100 * grid['predicted_risk']

# Create risk categories
grid['risk_category'] = pd.cut(
    grid['fire_risk_pct'],
    bins=[0, 20, 40, 60, 80, 100],
    labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
)

# ---------------------------------------
# STEP 5: Create visualizations
# ---------------------------------------
print("Creating fire risk visualizations...")

# 1. Fire Risk Heat Map
fig, ax = plt.subplots(figsize=(15, 12))

# Plot Florida boundary if available
if has_florida_shape:
    florida.boundary.plot(ax=ax, color='black', linewidth=1.5)

# Plot heat map
plot = grid.plot(
    column='fire_risk_pct',
    cmap='OrRd',
    linewidth=0.1,
    edgecolor='0.5',
    ax=ax,
    legend=True,
    vmin=0,
    vmax=100
)

# Fix the colorbar
cbar = plt.colorbar(plot.collections[0], ax=ax)
cbar.set_label('Fire Risk Score (0-100)', fontsize=12)

# Add fire locations
fl_gdf.plot(
    ax=ax,
    color='black',
    markersize=50,
    marker='*',
    label='Actual Fires'
)

# Add title and labels
if has_goes_data:
    ax.set_title('Florida Wildfire Risk Map with GOES-16 Data', fontsize=20)
else:
    ax.set_title('Florida Lightning-Induced Wildfire Risk Map', fontsize=20)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.legend(fontsize=12)

# Set bounds to Florida
ax.set_xlim([MIN_LON, MAX_LON])
ax.set_ylim([MIN_LAT, MAX_LAT])

# Save map
plt.savefig('florida_fire_risk_map.png', dpi=300, bbox_inches='tight')
print("Fire risk map saved to 'florida_fire_risk_map.png'")
plt.close()

# 2. Risk Category Map
fig, ax = plt.subplots(figsize=(15, 12))

# Plot Florida boundary if available
if has_florida_shape:
    florida.boundary.plot(ax=ax, color='black', linewidth=1.5)

# Define colors for risk categories
category_colors = {
    'Very Low': '#ffffcc',
    'Low': '#ffeda0',
    'Moderate': '#feb24c',
    'High': '#fc4e2a',
    'Very High': '#b10026'
}

# Create categorical map
for category, color in category_colors.items():
    subset = grid[grid['risk_category'] == category]
    if not subset.empty:
        subset.plot(
            color=color,
            ax=ax,
            label=category,
            edgecolor='0.5',
            linewidth=0.1
        )

# Add fire locations
fl_gdf.plot(
    ax=ax,
    color='black',
    markersize=50,
    marker='*',
    label='Actual Fires'
)

# Add legend, title and labels
ax.legend(fontsize=12)
ax.set_title('Florida Fire Risk Categories', fontsize=20)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

# Set bounds to Florida
ax.set_xlim([MIN_LON, MAX_LON])
ax.set_ylim([MIN_LAT, MAX_LAT])

# Save map
plt.savefig('florida_fire_risk_categories.png', dpi=300, bbox_inches='tight')
print("Risk categories map saved to 'florida_fire_risk_categories.png'")
plt.close()

# 3. Neural Network Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy', fontsize=16)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot loss
ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss', fontsize=16)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_training_curves.png', dpi=300)
print("Training curves saved to 'model_training_curves.png'")
plt.close()

# 4. Feature Importance
# We'll use the neural network weights to estimate feature importance
weights = model.layers[0].get_weights()[0]  # Get weights from first layer
importances = np.abs(weights).mean(axis=1)  # Average magnitude of weights per feature
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances / importances.sum()  # Normalize to sum to 1
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    feature_importance['Feature'],
    feature_importance['Importance'],
    color=['#ff9900', '#66cc33', '#3399ff', '#ffff00'] +
          (['#cc33ff', '#33cccc', '#cc6600'] if has_goes_data else [])
)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 0.01,
        f'{height:.2f}',
        ha='center', va='bottom', fontsize=12
    )

ax.set_title('Feature Importance from Neural Network', fontsize=18)
ax.set_ylabel('Relative Importance', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("Feature importance chart saved to 'feature_importance.png'")
plt.close()

# 5. Fire Incidents by Risk Category (Bar Chart)
risk_fire_counts = grid.groupby('risk_category')[['has_fire', 'total_fires']].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(risk_fire_counts['risk_category'], risk_fire_counts['total_fires'],
              color=[category_colors.get(cat, '#333333') for cat in risk_fire_counts['risk_category']])

# Add data labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12)

# Set title and labels
ax.set_title('Number of Fire Incidents by Risk Category', fontsize=18)
ax.set_xlabel('Risk Category', fontsize=14)
ax.set_ylabel('Number of Fires', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save bar chart
plt.tight_layout()
plt.savefig('fire_incidents_by_risk.png', dpi=300)
print("Bar chart saved to 'fire_incidents_by_risk.png'")
plt.close()

# 6. Combined Dashboard
fig = plt.figure(figsize=(20, 16))
grid_spec = fig.add_gridspec(2, 2)

# a. Risk Map (top left)
ax1 = fig.add_subplot(grid_spec[0, 0])
if has_florida_shape:
    florida.boundary.plot(ax=ax1, color='black', linewidth=1.5)
plot1 = grid.plot(
    column='fire_risk_pct',
    cmap='OrRd',
    linewidth=0.1,
    edgecolor='0.5',
    ax=ax1,
    legend=True,
    vmin=0,
    vmax=100
)
fl_gdf.plot(
    ax=ax1,
    color='black',
    markersize=30,
    marker='*',
    label='Actual Fires'
)
ax1.set_title('Fire Risk Map', fontsize=16)
ax1.set_xlim([MIN_LON, MAX_LON])
ax1.set_ylim([MIN_LAT, MAX_LAT])

# b. Model Learning Curves (top right)
ax2 = fig.add_subplot(grid_spec[0, 1])
ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Training Metrics', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# c. Feature Importance (bottom left)
ax3 = fig.add_subplot(grid_spec[1, 0])
bars = ax3.bar(
    feature_importance['Feature'],
    feature_importance['Importance'],
    color=['#ff9900', '#66cc33', '#3399ff', '#ffff00'] +
          (['#cc33ff', '#33cccc', '#cc6600'] if has_goes_data else [])
)
for bar in bars:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 0.01,
        f'{height:.2f}',
        ha='center', va='bottom', fontsize=10
    )
ax3.set_title('Feature Importance', fontsize=16)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# d. Fire Incidents by Risk (bottom right)
ax4 = fig.add_subplot(grid_spec[1, 1])
bars = ax4.bar(risk_fire_counts['risk_category'], risk_fire_counts['total_fires'],
               color=[category_colors.get(cat, '#333333') for cat in risk_fire_counts['risk_category']])
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)
ax4.set_title('Fire Incidents by Risk Category', fontsize=16)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('florida_fire_risk_dashboard.png', dpi=300)
print("Dashboard saved to 'florida_fire_risk_dashboard.png'")
plt.close()

# 7. Add visualization from shorter code (hot_r colormap)
print("Creating additional visualization from shorter code...")
fig, ax = plt.subplots(figsize=(10, 12))
grid.plot(column='fire_risk_pct', cmap='hot_r', legend=True, ax=ax, edgecolor='k', linewidth=0.1)

if has_florida_shape:
    florida.boundary.plot(ax=ax, color='black', linewidth=1)

plt.title("Florida Fire Risk Map (Predicted by Neural Network)", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('florida_fire_risk_hot_colormap.png', dpi=300)
print("Additional visualization saved to 'florida_fire_risk_hot_colormap.png'")
plt.close()

# Save results to GeoJSON
output_path = r"C:\Users\jason_vxe1\Downloads\FireNeuralNetwork\fire_risk_output.geojson"
grid.to_file(output_path, driver='GeoJSON')
print(f"Fire risk results saved to {output_path}")

# Print completion message
print("\nAll visualizations complete!")
print("The following files were created:")
print("1. florida_fire_risk_map.png - Heat map of fire risk scores")
print("2. florida_fire_risk_categories.png - Map of risk categories")
print("3. model_training_curves.png - Neural network learning curves")
print("4. feature_importance.png - Feature importance chart")
print("5. fire_incidents_by_risk.png - Bar chart of fire incidents by risk")
print("6. florida_fire_risk_dashboard.png - Combined dashboard view")
print("7. florida_fire_risk_hot_colormap.png - Additional visualization with hot_r colormap")
print(f"8. {output_path} - GeoJSON output with all data")
print("\nThese visualizations are ready for your presentation.")