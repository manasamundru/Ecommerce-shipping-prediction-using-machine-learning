from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import Normalizer
import pickle

app = Flask(__name__)

# Load your model and normalizer
model = pickle.load(open("rf_acc_68.pkl", "rb"))
Data_normalizer = pickle.load(open("normalizer.pkl", "rb"))

# Define a function to encode categorical variables if needed
def encode_categorical_variables(data):
    # Example: Encoding categorical variables into numeric representations
    data['Product_importance'] = {'low': 0, 'medium': 1, 'high': 2}.get(data['Product_importance'])
    data['Gender'] = {'male': 0, 'female': 1}.get(data['Gender'])
    data['Warehouse_block'] = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}.get(data['Warehouse_block'])
    data['Mode_of_shipment'] = {'Flight': 0, 'Ship': 1, 'Road': 2}.get(data['Mode_of_shipment'])
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Product_importance = request.form.get("Product_importance")
        Gender = request.form.get("Gender")
        Discount_offered = float(request.form.get("Discount_offered"))
        Weight_in_gms = float(request.form.get("Weight_in_gms"))
        Warehouse_block = request.form.get("Warehouse_block")
        Mode_of_shipment = request.form.get("Mode_of_shipment")
        Customer_care_calls = float(request.form.get("Customer_care_calls"))
        Customer_rating = float(request.form.get("Customer_rating"))
        Cost_of_product = float(request.form.get("Cost_of_product"))
        Prior_purchases = float(request.form.get("Prior_purchases"))

        # Prepare data for prediction
        # Convert categorical variables to numeric
        data = {
            'Product_importance': Product_importance,
            'Gender': Gender,
            'Discount_offered': Discount_offered,
            'Weight_in_gms': Weight_in_gms,
            'Warehouse_block': Warehouse_block,
            'Mode_of_shipment': Mode_of_shipment,
            'Customer_care_calls': Customer_care_calls,
            'Customer_rating': Customer_rating,
            'Cost_of_product': Cost_of_product,
            'Prior_purchases': Prior_purchases
        }
        data = encode_categorical_variables(data)

        # Convert data to numpy array and reshape for Normalizer
        preds = np.array(list(data.values())).reshape(1, -1)

        # Normalize data and make prediction
        normalized_preds = Data_normalizer.transform(preds)
        prediction = model.predict(normalized_preds)
        probability = model.predict_proba(normalized_preds)[0][1] * 100

        result_message = f'There is a {probability:.2f}% chance that your product will reach on time'

        # Render templates with prediction result
        return render_template("index.html", prediction=result_message)

if __name__ == '__main__':
    app.run(debug=True, port=4000)
