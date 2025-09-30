from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model data
with open("car_price_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]                # trained model
label_encoders = model_data["label_encoders"]
features = model_data["features"]

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/submit", methods=["POST"])
def submit():
    # Collect form data (JSON or form data)
    data = request.get_json()

    if not data:
        return {"error": "No data received"}, 400

    try:
        # Build DataFrame from input
        input_data = pd.DataFrame([{
            "Car_Name" : data.get("Car_Name"),
            "Year": int(data.get("Year")),
            "Present_Price": float(data.get("Present_Price")),
            "Kms_Driven": int(data.get("Kms_Driven")),
            "Fuel_Type": data.get("Fuel_Type"),
            "Seller_Type": data.get("Seller_Type"),
            "Transmission": data.get("Transmission"),
            "Owner": int(data.get("Owner"))
        }])
        # Apply label encoders to categorical columns
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = le.transform(input_data[col])

        # Ensure correct column order
        input_data = input_data[features]
        # Predict
        
        try : 
            predicted_price = model.predict(input_data)[0]
            print(predicted_price)
            return str(predicted_price)
        except Exception as e :
            print("error while predicting the price")
            return "error"
        return "never come to here"

    except Exception as e:
        return "internal server error here"

if __name__ == "__main__":
    app.run(debug=True,port=3000)
