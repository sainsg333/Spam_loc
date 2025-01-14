from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import requests
# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
app = Flask(__name__)
port = 5000
CORS(app)
@app.route('/detect', methods=['POST'])
def detect_spam():
    data = request.json
    print(data)
    message = data.get('message', '')
    location = data.get('location', 'Unknown')
    vectorized_msg = vectorizer.transform([message])
    prediction = model.predict(vectorized_msg)[0]

    result = {
        'message': message,
        'is_spam': bool(prediction),
    }
    print(result)
    return jsonify(result)

@app.route('/get-location', methods=['POST'])
def get_location():
    data = request.json
    email = data.get('email')
    phonenumbers = data.get('phone')
    ip = data.get('ip')
    try:
        token = "09a05661628cb2"  # Replace with your IPinfo API token
        response = requests.get(f"https://ipinfo.io/{ip}?token={token}")
        ip_data = response.json()
        location = ip_data.get('loc', '0,0').split(',')  # Latitude and Longitude
        return {
            'city': ip_data.get('city', 'Unknown'),
            'region': ip_data.get('region', 'Unknown'),
            'country_name': ip_data.get('country', 'Unknown'),
            'latitude': float(location[0]) if len(location) > 1 else 0.0,
            'longitude': float(location[1]) if len(location) > 1 else 0.0,
        }
    except Exception as e:
        return {'error': 'Failed to fetch IP location', 'details': str(e)}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,port=port)
