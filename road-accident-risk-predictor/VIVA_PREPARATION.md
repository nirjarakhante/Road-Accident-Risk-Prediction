# 🎓 Road Accident Risk Prediction — Viva & Seminar Guide

Yeh guide aapko viva, seminar, aur project presentation mein har question ka confident aur accurate answer dene ke liye banayi gayi hai.

---

## 1. 🎤 1-Minute Project Pitch (Examiner: "Introduce your project")

> *"Sir/Madam, my project is **Road Accident Risk Prediction using Machine Learning**. 
> Most conventional research predicts post-crash severity after an accident has already taken place. Our system takes a **proactive approach**: we analyze historical crash data, weather conditions, time patterns, and road infrastructure (like junctions, traffic signals, and crossings) to classify road segments into **Low**, **Medium**, or **High Risk (Red Alert)** zones. 
> This allows traffic authorities to plan proactive interventions like better signage, automated speed enforcement, or active patrolling before fatal collisions occur. The project is implemented in **Python** using algorithms like **XGBoost** and **Random Forest**, with an interactive **Streamlit and Folium** geospatial frontend."*

---

## 2. ❓ Top 8 Viva Questions & Exact Answers

### Q1: Why predict "Location Risk" instead of "Accident Severity"?
* **Answer**: *"Predicting severity of an ongoing crash is reactive. Predicting location risk is **proactive** — it helps urban planners and commuters know which road segments are inherently hazardous under specific weather and traffic conditions."*

---

### Q2: What dataset did you use, and how did you handle its size?
* **Answer**: *"We based our schema on the official **Kaggle US Accidents Dataset** (covering multi-year nationwide crashes). Since the full CSV is over 7 GB, we engineered a scalable sampling and spatial gridding pipeline (~0.05° grid cells, approx 5 km squares) to aggregate accident points into continuous road corridors."*

---

### Q3: How did you create the "Low", "Medium", and "High" target labels?
* **Answer**: *"We formulated a **Composite Risk Score** combining both frequency and severity:
  $$\text{RiskScore} = 0.6 \times \text{Normalized Accident Count} + 0.4 \times \text{Normalized Severity}$$
  Then we applied quantile thresholding (top 15% scored as High Risk/Red Alert, next 30% Medium, and remaining 55% Low Risk). This reflects real-world road networks where severe blackspots are a minority."*

---

### Q4: 🌟 STAR QUESTION: What is "Target Leakage" and how did you fix it?
*(Examiners LOVE this question — it shows real engineering honesty!)*
* **Answer**: *"In our initial prototype, our model achieved a suspicious **0.99 Macro F1 score**. Upon auditing the feature matrix, we discovered that `AccidentCount` and `AvgSeverity` were accidentally included as input features. Since those two numbers were used to generate the target label itself, the model was essentially memorizing the answer.
  We fixed this by **completely removing target-derived features**, retaining only independent physical attributes: Junction, Traffic Signal, Crossing, Rush Hour, Weekend, Visibility, Precipitation, and Bad Weather. The score dropped to an honest **~0.49 Macro F1**, which proves genuine generalization without data contamination."*

---

### Q5: Why is your Macro F1 score ~0.49? Isn't that low?
* **Answer**: *"No sir. In a 3-class problem with severe class imbalance (where High risk is only 15%), a random guess yields ~0.33 Macro F1. An honest **0.49 Macro F1** using only independent environmental and road features is a realistic result for spatial accident prediction. High accuracy like 95%+ in such problems is almost always an indicator of data leakage or overfitting."*

---

### Q6: Why did you use SMOTE instead of simple oversampling?
* **Answer**: *"High-risk road segments are the minority class. Simple random oversampling just duplicates existing minority rows, causing tree models like Random Forest and XGBoost to overfit. **SMOTE (Synthetic Minority Over-sampling Technique)** creates synthetic, interpolated data points between nearest neighbors in feature space, helping the classifier learn smoother decision boundaries."*

---

### Q7: Which algorithm performed best and why?
* **Answer**: *"We compared 5 models: **Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, and XGBoost**. **XGBoost** achieved the highest test Macro F1 (~0.49) because gradient-boosted decision trees effectively capture complex non-linear interactions (e.g. how wet pavement interacts with blind highway junctions during night rush hours)."*

---

### Q8: How does the map visualization work?
* **Answer**: *"We used **Folium** wrapped in **Streamlit-Folium**. Each road segment's latitude and longitude are plotted as color-coded circular markers (Red = High Risk, Orange = Medium, Green = Low). Clicking any marker opens an interactive HTML popup detailing past crashes, severity, and infrastructure details."*

---

## 3. 📂 Code Architecture Cheat Sheet (File by File)

1. **`src/preprocessing.py`**:
   - Cleans missing values.
   - Snaps lat/lng to ~5km grid cells (`GridLat`, `GridLng`).
   - Computes infrastructure percentages and creates the `Risk_Level` label.
2. **`src/train_model.py`**:
   - Splits data into Train & Test (80/20 stratified).
   - Applies `SMOTE` on the training set.
   - Evaluates 5 models on Macro F1.
   - Saves `risk_model.pkl`, `scaler.pkl`, and `label_encoder.pkl`.
3. **`app/streamlit_app.py`**:
   - **Tab 1 (City Risk Map)**: Filter by city, view red alert accident hotspots on an interactive map.
   - **Tab 2 (Predict Road Risk)**: Choose road layout (junction/signal), weather (rain/fog/clear), and rush hour -> predicts risk level with confidence bar chart.
   - **Tab 3 (Model Comparison)**: Shows the F1 score leaderboard and explains the leakage fix.
