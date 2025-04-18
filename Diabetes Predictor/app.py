
from flask import Flask, request, render_template
import os
import joblib  

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        fn = "dia.pkl"
        if os.path.exists(fn):
            try:
                # Use joblib to load the model
                model = joblib.load(fn)

                # Validate and convert inputs
                try:
                    fs = float(request.form["fs"])
                    fu = int(request.form["fu"])
                except ValueError:
                    return render_template("home.html", msg="Invalid input. Please enter numeric values.")

                # Prepare input data
                if fu == 1:
                    d = [[fs, 0]]
                else:
                    d = [[fs, 1]]

                # Make prediction
                result = model.predict(d)
                msg = str(result[0])
                return render_template("home.html", msg=msg)

            except Exception as e:
                return render_template("home.html", msg=f"Error loading model: {str(e)}")
        else:
            msg = f"{fn} does not exist."
            return render_template("home.html", msg=msg)
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)