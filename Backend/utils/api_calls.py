import requests

# Your WAQI API Token
API_TOKEN = "0bba4057a089120a997f46c07fa7512f3f595ed7"

def get_air_quality(city):
    """
    Fetch real-time air quality data for a given city from the WAQI API.
    
    Args:
        city (str): The city name to fetch AQI data for.

    Returns:
        dict: A dictionary containing AQI and pollutant data.
    """
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

            # Displaying results
            print(f"City: {city_name}")
            print(f"AQI: {aqi}")
            print(f"PM2.5: {pm25} µg/m³")
            print(f"PM10: {pm10} µg/m³")
            print(f"NO2: {no2} µg/m³")
            print(f"Last Updated: {last_updated}")
        else:
            print("Failed to fetch data. API Response:", data)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Get city name from user input
city = input("Enter the city name: ").strip()  # Removing extra spaces around input
get_air_quality(city)




