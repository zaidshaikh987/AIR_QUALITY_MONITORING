from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Your WAQI API Token
API_TOKEN = "0bba4057a089120a997f46c07fa7512f3f595ed7"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_aqi', methods=['POST'])
def get_aqi():
    city = request.form['city']  # Get city from the form data
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    
    try:
        # Fetch data from the API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues (4xx/5xx)

        # Parse JSON response
        data = response.json()
        if data['status'] == 'ok':
            aqi = data['data']['aqi']
            pm25 = data['data']['iaqi'].get('pm25', {}).get('v', 'N/A')  # Handle missing key
            pm10 = data['data']['iaqi'].get('pm10', {}).get('v', 'N/A')  # Handle missing key
            no2 = data['data']['iaqi'].get('no2', {}).get('v', 'N/A')    # Handle missing key
            city_name = data['data']['city']['name']
            last_updated = data['data']['time']['s']
            
            # Return data as JSON
            return jsonify({
                'status': 'success',
                'city': city_name,
                'aqi': aqi,
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'last_updated': last_updated
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to fetch data from API'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

