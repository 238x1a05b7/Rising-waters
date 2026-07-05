from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Verify pathways and load model and scaler binaries
model_path = os.path.join('models', 'flood_model.joblib')
scaler_path = os.path.join('models', 'scaler.joblib')

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    print("Warning: Model files missing from your project directories!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Map dashboard inputs into separate float values
        annual_rf = float(data.get('Annual_Rainfall', 0))
        cloud_vis = float(data.get('Cloud_Cover', 0))  # Mapped to fulfill Cloud_Visibility
        avg_temp  = float(data.get('Temperature', 0))
        humidity  = float(data.get('Humidity', 0))
        avg_june  = float(data.get('Avg_June', 0))
        risk_idx  = float(data.get('Flood_Risk_Index', 0))
        
        # 2. Add seasonal rainfall components as the 7th structural feature
        # (Combining or picking your main monsoon period column to satisfy the shape requirement)
        jun_sep   = float(data.get('Jun_Sep', 0))

        # 3. Re-arrange values array structure into model training sequence order (exactly 7 features)
        # Note: If your notebook has a different specific order, shuffle these items to match it.
        feature_vector = [
            annual_rf,    # Feature 1
            cloud_vis,    # Feature 2 ('Cloud_Visibility')
            avg_temp,     # Feature 3
            humidity,     # Feature 4
            avg_june,     # Feature 5
            risk_idx,     # Feature 6
            jun_sep       # Feature 7 (Completes the structural 7-feature requirement)
        ]
        
        # 4. Format and scale vector matrices 
        input_data = np.array([feature_vector])
        scaled_data = scaler.transform(input_data)
        
       # 5. Generate system calculations
        prediction_result = model.predict(scaled_data)
        
       
        raw_val = str(prediction_result).strip("[]'\" ")
        
        
        if raw_val == "1":
            final_score = "⚠️ High Chance"
        else:
            final_score = "✅ Low Chance"
        
        return jsonify({'success': True, 'prediction': final_score})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f"Prediction error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)