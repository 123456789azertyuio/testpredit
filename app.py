from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load("modele_agences.pkl")
encoder = joblib.load("encoder_agences.pkl")

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            file.save(os.path.join(UPLOAD_FOLDER, "base.xlsx"))
            return redirect(url_for("admin"))
    return render_template("admin.html")

@app.route("/user", methods=["GET", "POST"])
def user():
    prediction_table = None
    path = os.path.join(UPLOAD_FOLDER, "base.xlsx")
    if not os.path.exists(path):
        return "Aucune base de données n’a été importée par l’admin."

    df = pd.read_excel(path)

    if request.method == "POST":
        try:
            vmin = float(request.form["vmin"])
            vmax = float(request.form["vmax"])
        except ValueError:
            return "Entrer des valeurs numériques valides."

        df["vmin"] = vmin
        df["vmax"] = vmax

        cat_features = ["Région", "Réseau", "Code Agence", "Classification", 
                        "Profil PTF", "Code PTF", "Marché"]
        num_features = ["Prod N-2", "Prod N-1", "Prod N", "RAF", "Coef.", "vmin", "vmax"]

        encoded = encoder.transform(df[cat_features])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_features))

        X = pd.concat([df[num_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        y_pred = model.predict(X)
        results = pd.DataFrame(y_pred, columns=["Moyenne Annuelle", "Min", "Max", "Objectif"])
        prediction_table = pd.concat([df.reset_index(drop=True), results], axis=1)


    return render_template("user.html", tables=[prediction_table.to_html(classes="table table-bordered", index=False)] if prediction_table is not None else None)

if __name__ == "__main__":
    app.run(debug=True)
