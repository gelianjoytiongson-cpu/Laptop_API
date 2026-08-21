"""
app.py
------
Flask backend for the Flutter laptop-price-prediction app.
Loads laptop_price_model.pkl (produced by train_and_export.py) and
exposes:

    GET  /           -> root status check (for a quick browser test)
    GET  /health      -> simple liveness check
    GET  /options     -> valid dropdown values for every categorical field
    POST /predict     -> predicted price for a given laptop spec

Run with:
    python app.py
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(_name_)
CORS(app)  # allow the Flutter app (different origin/device) to call this API

BUNDLE_PATH = 'laptop_price_model.pkl'
bundle = joblib.load(BUNDLE_PATH)

model = bundle['model']
scaler = bundle['scaler']
le_proc = bundle['le_proc']
le_gpu = bundle['le_gpu']
le_panel = bundle['le_panel']
feature_columns = bundle['feature_columns']
numerical_cols = bundle['numerical_cols']
defaults = bundle['defaults']
options = bundle['options']

REQUIRED_FIELDS = [
    'brand', 'screen_size', 'screen_width', 'screen_height', 'screen_panel',
    'operating_system', 'cpu_brand', 'processor_model', 'ram_gb',
    'storage_gb', 'storage_type', 'gpu_model',
]


def build_feature_row(payload):
    """Turns a JSON payload into a single-row DataFrame matching the
    training feature layout, mirroring predict_price() from the notebook."""
    row = dict(defaults)  # start from median/mode defaults for every column

    row['Screen_Size'] = float(payload['screen_size'])
    row['Screen_Width'] = float(payload['screen_width'])
    row['Screen_Height'] = float(payload['screen_height'])
    row['RAM_GB'] = float(payload['ram_gb'])
    row['Storage_GB'] = float(payload['storage_gb'])

    row['Processor_Model_Encoded'] = le_proc.transform([payload['processor_model']])[0]
    row['GPU_Model_Encoded'] = le_gpu.transform([payload['gpu_model']])[0]
    row['Screen_Panel_Encoded'] = le_panel.transform([payload['screen_panel']])[0]

    dummy_choices = [
        (payload['brand'], '^Brand_'),
        (payload['operating_system'], '^Operating System_'),
        (payload['cpu_brand'], '^CPU_Brand_'),
        (payload['storage_type'], '^Storage_Type_'),
    ]
    all_cols_series = pd.Index(feature_columns)
    for choice, prefix in dummy_choices:
        matching_cols = all_cols_series[all_cols_series.str.match(prefix)]
        for col in matching_cols:
            row[col] = 0
        target_col = f"{prefix.lstrip('^')}{choice}"
        if target_col in feature_columns:
            row[target_col] = 1

    input_df = pd.DataFrame([row])[feature_columns]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    return input_df


def validate_payload(payload):
    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, '')]
    if missing:
        return f"Missing field(s): {', '.join(missing)}"

    checks = [
        ('brand', options['brand']),
        ('screen_panel', options['screen_panel']),
        ('operating_system', options['operating_system']),
        ('cpu_brand', options['cpu_brand']),
        ('processor_model', options['processor_model']),
        ('storage_type', options['storage_type']),
        ('gpu_model', options['gpu_model']),
    ]
    for field, valid_values in checks:
        if payload[field] not in valid_values:
            return f"Invalid value for '{field}': {payload[field]!r}. Must be one of {valid_values}"

    for numeric_field in ['screen_size', 'screen_width', 'screen_height', 'ram_gb', 'storage_gb']:
        try:
            float(payload[numeric_field])
        except (TypeError, ValueError):
            return f"'{numeric_field}' must be a number"

    return None


@app.route('/', methods=['GET'])
def index():
    """Root status check — handy for confirming the deploy is live
    straight from a browser, without needing to know a specific route."""
    return jsonify({
        'status': 'online',
        'service': 'Laptop Price Prediction API',
        'endpoints': ['/health', '/options', '/predict'],
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/options', methods=['GET'])
def get_options():
    """Returns valid dropdown values so the Flutter UI can populate its pickers."""
    return jsonify(options)


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'Request body must be JSON'}), 400

    error = validate_payload(payload)
    if error:
        return jsonify({'error': error}), 400

    try:
        input_df = build_feature_row(payload)
        predicted_price = float(model.predict(input_df)[0])
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'predicted_price': round(predicted_price, 2)})


if _name_ == '_main_':
    # host='0.0.0.0' so a physical phone / emulator on the same network
    # (or an Android emulator via 10.0.2.2) can reach this server.
    app.run(host='0.0.0.0', port=5000, debug=True)