import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
file_path = 'C:\FinalYearProject-IDS\iot23_combined_new.csv' # Make sure this is your file's name

try:
    col_names = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
        'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts',
        'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'label'
    ]
    df = pd.read_csv(file_path, sep='\t', header=None, names=col_names, comment='#', on_bad_lines='skip')
    print(f"File loaded successfully! Found {len(df)} rows.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --- 2. Data Cleaning and Feature Engineering (IMPROVED) ---

# Replace placeholders with NaN so we can work with them
df.replace(['-', '(empty)'], np.nan, inplace=True)

# Drop columns that are definitely not useful
columns_to_drop = [
    'ts', 'uid', 'id.orig_h', 'id.resp_h',  # Identifiers
    'service', 'local_orig', 'local_resp', 'missed_bytes' # Mostly empty/not useful
]
df.drop(columns=columns_to_drop, inplace=True)

# --- NEW: Intelligent Handling of Missing Values ---
# For numerical columns, fill missing values with 0
# For categorical/object columns, fill missing values with the string 'missing'
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna('missing', inplace=True)
    else:
        df[col].fillna(0, inplace=True)

print(f"\nData after cleaning: {len(df)} rows remaining.")
if len(df) == 0:
    print("Error: All data was still removed. Please check the input file for corruption.")
    exit()

# --- 3. Preprocessing for the RNN Model ---

# Convert all data types to be safe
df = df.astype({'proto': str, 'conn_state': str, 'history': str, 'label': str,
                'id.orig_p': float, 'id.resp_p': float, 'duration': float,
                'orig_bytes': float, 'resp_bytes': float, 'orig_pkts': float,
                'orig_ip_bytes': float, 'resp_pkts': float, 'resp_ip_bytes': float})


# Identify features
categorical_features = ['proto', 'conn_state', 'history']
numerical_features = [col for col in df.columns if col not in categorical_features and col != 'label']

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Encode the target label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['label'])
y_categorical = to_categorical(y_encoded)

# Separate features (X) and target (y)
X = df.drop(columns=['label'])

# Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

print("\nPreprocessed Data Head:")
print(X.head())

# --- 4. Prepare Data for RNN ---

X_train, X_test, y_train, y_test = train_test_split(X.values, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"\nTraining data shape: {X_train_reshaped.shape}")

# --- 5. Build and Train the RNN Model ---

num_classes = y_categorical.shape[1]
model = Sequential([
    LSTM(128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\n--- Starting Model Training ---")
history = model.fit(
    X_train_reshaped, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)

# --- 6. Evaluate and Save ---
print("\n--- Evaluating Model Performance ---")
loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

model.save('sdn_idps_rnn_model.h5')
print("\nModel saved successfully as 'sdn_idps_rnn_model.h5'")
