
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import folium
import os

app = Flask(__name__)

def train_model():
    import os

    if not os.path.exists("static"):
        os.makedirs("static")

    if not os.path.exists("templates"):
        os.makedirs("templates")

    df = pd.read_csv("voicequality_data.csv")
    
    # Set modern style
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12    

    # Operator popularity
    operator_counts = df['operator'].value_counts()

    plt.figure()
    sns.barplot(x=operator_counts.values,
                y=operator_counts.index,
                palette="viridis")

    plt.title("Operator Popularity (User Count)")
    plt.xlabel("Number of Users")
    plt.ylabel("Operator")
    plt.tight_layout()
    plt.savefig("static/operator_popularity.png")
    plt.close()

    plt.figure()

    operator_counts.plot.pie(autopct='%1.1f%%',
                            cmap="Set2")
    plt.title("Operator Market Share")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("static/operator_share.png")
    plt.close()

    df = df[df["rating"].notnull()]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna()

    df["target"] = df["calldrop_category"].apply(
        lambda x: 1 if x == "Call Dropped" else 0
    )

    le_operator = LabelEncoder()
    le_network = LabelEncoder()
    le_travel = LabelEncoder()

    df["operator_enc"] = le_operator.fit_transform(df["operator"])
    df["network_enc"] = le_network.fit_transform(df["network_type"])
    df["travel_enc"] = le_travel.fit_transform(df["inout_travelling"])

    X = df[["rating","operator_enc","network_enc","travel_enc","month"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # Feature Importance Plot
    importances = model.feature_importances_
    features = X.columns

    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure()
    sns.barplot(x="Importance",
                y="Feature",
                data=feat_df,
                palette="coolwarm")

    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig("static/feature_importance.png")
    plt.close()

    # Calculate drop rate properly
    df['drop_flag'] = df['calldrop_category'].apply(
        lambda x: 1 if x == "Call Dropped" else 0
    )

    state_drop = df.groupby('state_name')['drop_flag'].mean()
    state_drop = state_drop.sort_values(ascending=True)

    plt.figure(figsize=(8,6))
    sns.barplot(x=state_drop.values,
                y=state_drop.index,
                palette="mako")

    plt.title("State-wise Call Drop Rate")
    plt.xlabel("Call Drop Probability")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig("static/state_drop.png")
    plt.close()

    import plotly.graph_objects as go

# Overall Drop Rate

    # Create drop flag first
    df['drop_flag'] = df['calldrop_category'].apply(
        lambda x: 1 if x == "Call Dropped" else 0
    )

    # Calculate overall drop rate
    overall_drop_rate = round(df['drop_flag'].mean() * 100, 2)
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall_drop_rate,
    number={'suffix': "%", 'font': {'size': 40}},
    title={'text': "Overall Call Drop Rate (%)", 'font': {'size': 20}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkred"},
        'steps': [
            {'range': [0, 20], 'color': "#2ecc71"},
            {'range': [20, 40], 'color': "#f1c40f"},
            {'range': [40, 100], 'color': "#e74c3c"}
        ]
    }
    ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )

    fig.write_html("static/drop_gauge.html", include_plotlyjs='cdn')

    # Map Visualization — dark theme
    m = folium.Map(
        location=[20.59, 78.96],
        zoom_start=5,
        tiles="CartoDB dark_matter"
    )

    sample = df.sample(min(300, len(df)))
    for _, row in sample.iterrows():
        if row["latitude"] != -1 and row["longitude"] != -1:
            dropped = row["target"] == 1
            color   = "#e74c3c" if dropped else "#2ecc71"
            status  = "📵 Call Dropped" if dropped else "✅ Connected"
            popup_html = f"""
              <div style='font-family:Inter,sans-serif;font-size:13px;min-width:160px'>
                <b style='color:{color}'>{status}</b><br>
                🏢 Operator: <b>{row.get('operator','—')}</b><br>
                📶 Network: <b>{row.get('network_type','—')}</b><br>
                🌍 State: <b>{row.get('state_name','—')}</b><br>
                ⭐ Rating: <b>{row.get('rating','—')}</b>
              </div>"""
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5 if dropped else 4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=status
            ).add_to(m)

    m.save("templates/map.html")

    encoders = {
        "operator": le_operator,
        "network": le_network,
        "travel": le_travel
    }

    return model, encoders, acc

model, encoders, accuracy = train_model()

@app.route("/")
def home():
    return render_template("index.html", accuracy=round(accuracy*100,2))

@app.route("/predict", methods=["POST"])
def predict():
    rating = int(request.form["rating"])
    month = int(request.form["month"])
    operator = request.form["operator"]
    network = request.form["network"]
    travel = request.form["travel"]

    operator_enc = encoders["operator"].transform([operator])[0]
    network_enc = encoders["network"].transform([network])[0]
    travel_enc = encoders["travel"].transform([travel])[0]

    input_data = [[rating, operator_enc, network_enc, travel_enc, month]]
    prob = model.predict_proba(input_data)[0][1]  # probability of drop

    if prob > 0.5:
        result = "High Risk of Call Drop"
    elif prob > 0.25:
        result = "Moderate Risk"
    else:
        result = "Low Risk - Good Call Quality"

    print("Input:", input_data)
    print("Drop Probability:", prob)

    return render_template(
        "index.html",
        prediction_text=result,
        accuracy=round(accuracy*100,2),
        prob=round(prob*100,2)
    )

@app.route("/map")
def map_view():
    return render_template("map.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    from flask import jsonify
    try:
        rating  = int(request.form["rating"])
        month   = int(request.form["month"])
        operator = request.form["operator"]
        network  = request.form["network"]
        travel   = request.form["travel"]

        operator_enc = encoders["operator"].transform([operator])[0]
        network_enc  = encoders["network"].transform([network])[0]
        travel_enc   = encoders["travel"].transform([travel])[0]

        input_data = [[rating, operator_enc, network_enc, travel_enc, month]]
        prob = model.predict_proba(input_data)[0][1]

        if prob > 0.5:
            result = "High Risk of Call Drop"
            level  = "high"
        elif prob > 0.25:
            result = "Moderate Risk"
            level  = "mod"
        else:
            result = "Low Risk – Good Call Quality"
            level  = "low"

        return jsonify({"result": result, "level": level, "prob": round(prob * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
