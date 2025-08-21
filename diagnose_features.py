import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load and Prepare Data (Same as before) ---
file_path = 'C:\FinalYearProject-IDS\iot23_combined_new.csv' # Make sure this is your file's name
col_names = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
    'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts',
    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'label'
]
try:
    df = pd.read_csv(file_path, sep='\t', header=None, names=col_names, comment='#', on_bad_lines='skip')
    print(f"File loaded successfully! Found {len(df)} rows.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

df.replace(['-', '(empty)'], np.nan, inplace=True)
columns_to_drop = [
    'ts', 'uid', 'id.orig_h', 'id.resp_h', 'service', 
    'local_orig', 'local_resp', 'missed_bytes'
]
df.drop(columns=columns_to_drop, inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna('missing', inplace=True)
    else:
        df[col].fillna(0, inplace=True)

# --- 2. Encode ALL columns for the Random Forest ---
# We need to convert everything to numbers to check for importance
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop('label', axis=1)
y = df_encoded['label']

# --- 3. Train the Random Forest and Get Feature Importances ---
print("\nTraining a Random Forest to find important features...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# --- 4. Create and Display the Feature Importance Plot ---
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Feature Importance Results ---")
print(feature_importances)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance for Predicting Traffic Type')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
# Save the plot as an image file
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")
plt.show()
