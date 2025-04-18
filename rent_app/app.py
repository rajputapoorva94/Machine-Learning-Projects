from flask import Flask, request, render_template
from pickle import load
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        fn = "re.pkl" 
        if os.path.exists(fn):
            try:
                # Load the model safely
                with open(fn, "rb") as f:
                    model = load(f)
            except Exception as e:
                return render_template("home.html", msg=f"Error loading model: {str(e)}")

            # Get user input
            try:
                bhk = int(request.form["bhk"])
                size = float(request.form["size"])
                fs = int(request.form["fs"])
                bathroom = int(request.form["bathroom"])
            except ValueError:
                return render_template("home.html", msg="Invalid input. Please enter numeric values.")

            # Prepare input data
            if fs == 1:
                d = [[bhk, size, bathroom, 1, 0]]
            else:
                d = [[bhk, size, bathroom, 0, 1]]

            # Make prediction
            try:
                rent = model.predict(d)
                msg = "Rent = " + str(round(rent[0], 2)) + "K"
            except Exception as e:
                msg = f"Error making prediction: {str(e)}"

            return render_template("home.html", msg=msg)
        else:
            return render_template("home.html", msg=f"Model file '{fn}' not found.")
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)