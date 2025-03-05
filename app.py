from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_joblib_heart')

# Example dietary plan
diet_plan = [
    "Eat a variety of fruits and vegetables.",
    "Choose whole grains over refined grains.",
    "Limit saturated fat and avoid trans fat.",
    "Eat lean proteins such as fish, poultry, and legumes.",
    "Reduce salt intake to control blood pressure.",
    "Avoid sugary drinks and limit added sugars."
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract input data from the form
            data = [
                int(request.form['age']),
                int(request.form['sex']),
                int(request.form['cp']),
                int(request.form['trestbps']),
                int(request.form['chol']),
                int(request.form['fbs']),
                int(request.form['restecg']),
                int(request.form['thalach']),
                int(request.form['exang']),
                float(request.form['oldpeak']),
                int(request.form['slope']),
                int(request.form['ca']),
                int(request.form['thal'])
            ]

            # Make prediction
            prediction = model.predict([data])

            # Interpret result
            if prediction[0] == 1:
                result = "Possibility of heart disease"
                return render_template('result.html', result=result, diet_plan=diet_plan)
            else:
                result = "You are normal. No heart disease detected."
                return render_template('result.html', result=result)

        except Exception as e:
            print(f"Error: {e}")
            result = "An error occurred. Please check the inputs and try again."
            return render_template('result.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
