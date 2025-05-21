from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model and location frequency map
model = joblib.load(os.path.join('models', 'final_lgb_model.pkl'))
loc_freq_map = joblib.load(os.path.join('models', 'location_freq_encoding.pkl'))

@app.route('/')
def index():
    return render_template('index.html', location_list=sorted(loc_freq_map.keys()))

def format_price(price):
    if price >= 1_00_00_000:
        return f"Rs. {price/1_00_00_000:.2f} Crores"
    elif price >= 1_00_000:
        return f"Rs. {price/1_00_000:.2f} Lakhs"
    else:
        return f"Rs. {price:,.0f}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        area = float(request.form['area'])
        location = request.form['location']

        location_freq = loc_freq_map.get(location, 0)
        bb = bedrooms * bathrooms
        ba = bedrooms * area
        ba2 = bathrooms * area

        input_df = pd.DataFrame([{
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Area_Marla': area,
            'Bedrooms Bathrooms': bb,
            'Bedrooms Area_Marla': ba,
            'Bathrooms Area_Marla': ba2,
            'Primary_Location_FreqEnc': location_freq
        }])

        predicted_price = model.predict(input_df)[0]
        predicted_price = round(predicted_price)

        # Format price
        formatted_price = format_price(predicted_price)

        # Price range (±10%)
        lower = round(predicted_price * 0.9)
        upper = round(predicted_price * 1.1)
        formatted_lower = format_price(lower)
        formatted_upper = format_price(upper)

        prediction_text = (
            f"Estimated Price: {formatted_price} "
            f"(Range: {formatted_lower} - {formatted_upper})"
        )

        return render_template('index.html',
                               location_list=sorted(loc_freq_map.keys()),
                               prediction=prediction_text)

    except Exception as e:
        return render_template('index.html',
                               location_list=sorted(loc_freq_map.keys()),
                               prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
