import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR  # ADDED Support Vector Regressor (SVR)

# Streamlit GUI Setup
st.set_page_config(page_title="Construction Cost Estimator", page_icon="🏗️", layout="centered")

# Display Logo in Top Left Corner
st.image("https://i.postimg.cc/fLcqQZ2q/RAK-4-DIGITAL-SCREEN-GREY-BACKG.png", width=150)

# --------------------------- Load Existing Model & Data ---------------------------

# Define output directory
output_dir = "C:/Users/Serge/Desktop/Construction Model Data/Code Output"
file_path = os.path.join(output_dir, "Structured_Master_Construction_Costs.xlsx")

# Load structured dataset
df = pd.read_excel(file_path, sheet_name=None)
df_all = pd.concat(df.values(), ignore_index=True)

# Compute Additional Features
df_all["Cost per Square Meter"] = df_all["Total Cost (USD)"] / df_all["Quantity"]
df_all["Labor-to-Material Ratio"] = df_all["Total Cost (USD)"] / df_all["Unit Price (USD)"]
df_all["Cost per Floor"] = df_all["Total Cost (USD)"] / df_all["Quantity"]
df_all["Inflation Adjustment"] = df_all["Total Cost (USD)"] * 1.05  # Assuming 5% yearly inflation

# Aggregate data by project and cost category
df_pivot = df_all.pivot_table(
    values=['Total Cost (USD)', 'Cost per Square Meter', 'Labor-to-Material Ratio'],
    index=['Project'],
    columns=['Cost Category'],
    aggfunc='sum',
    fill_value=0
).reset_index()

# Flatten Multi-Index Columns
df_pivot.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_pivot.columns]

# Ensure 'Project' column exists
df_pivot = df_pivot.rename(columns={'Project_': 'Project'})

# Define Features & Target
if 'Project' in df_pivot.columns:
    X = df_pivot.drop(columns=['Project'])
else:
    st.error("⚠️ Error: 'Project' column is missing in df_pivot. Please check the dataset.")

y = df_pivot.drop(columns=['Project']).sum(axis=1)  # Predicting total cost

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------- Defining Best Model to Use ---------------------------

# Define Multiple Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42),
    "Support Vector Regressor (SVR)": SVR(kernel="rbf")
}

# Store Performance Metrics
mae_scores, rmse_scores, r2_scores = {}, {}, {}

# Train & Evaluate Each Model
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae_scores[model_name] = mean_absolute_error(y_test, y_pred)
    rmse_scores[model_name] = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_scores[model_name] = model.score(X_test_scaled, y_test)

# Select Best Model Based on RMSE
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]

# Save Best Model & Scaler
model_path = os.path.join(output_dir, "best_construction_cost_model.pkl")
scaler_path = os.path.join(output_dir, "scaler.pkl")
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

# --------------------------- Model Performance UI Section ---------------------------

with st.expander("📊 Model Comparison Results (Click to Expand/Collapse)"):
    st.write("### 📊 Model Comparison Results")

    # Convert model performance metrics to a DataFrame
    model_results_df = pd.DataFrame({
        "Model": list(mae_scores.keys()),
        "MAE": [f"{mae_scores[m]:,.3f}" for m in mae_scores.keys()],
        "RMSE": [f"{rmse_scores[m]:,.3f}" for m in mae_scores.keys()],
        "R² Score": [f"{r2_scores[m]:.4f}" for m in mae_scores.keys()]
    })

    # Display table
    st.table(model_results_df)

    # **Show the selected best model**
    st.success(f"✅ **The best model selected: {best_model_name}**")



# --------------------------- Streamlit GUI for Cost Estimation ---------------------------

# Load trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Streamlit GUI Setup
#st.set_page_config(page_title="Construction Cost Estimator", page_icon="🏗️", layout="centered")

st.title("🏗️ Construction Cost Estimator")
st.write("### Enter project details below to estimate the total construction cost.")

# User Input Fields
project_type = st.selectbox("🏗️ Select Project Type", ["Residential", "Industrial", "Commercial", "Infrastructure", "Education", "Healthcare"])
regional_cost_index = {
    "Dubai": 1.30,
    "Abu Dhabi": 1.25,
    "Riyadh": 1.20,
    "Jeddah": 1.15,
    "Doha": 1.22,
    "Kuwait City": 1.18,
    "Cairo": 0.85,
    "Istanbul": 0.90,
    "London": 1.50,
    "New York": 1.70
}

# **Initialize session state for the cost factor if not already set**
if "location_factor" not in st.session_state:
    st.session_state["location_factor"] = 1.0  # Default value

# **Location Selection**
selected_region = st.selectbox("🌍 Select Project Location", list(regional_cost_index.keys()))

# **Auto-Adjust Cost Factor Based on Selection**
if st.session_state["location_factor"] != regional_cost_index[selected_region]:  
    st.session_state["location_factor"] = regional_cost_index[selected_region]  # Set new value

# **Fine-Tune Cost Factor Slider (Pre-Filled with Selected Region's Value)**
location_factor = st.slider(
    "📍 Fine-Tune Cost Factor/Pirce Index (Manual Adjustment)", 
    min_value=0.5, 
    max_value=2.0, 
    value=st.session_state["location_factor"]
)
material_quality = st.selectbox("🛠️ Material Quality", ["Standard", "Premium", "Luxury"])
project_size = st.number_input("📏 Enter Project Size (sqm)", min_value=10, max_value=100000, value=1000)
num_floors = st.number_input("🏢 Enter Number of Floors", min_value=1, max_value=100, value=5)
timeline = st.number_input("⏳ Enter Project Timeline (months)", min_value=1, max_value=60, value=12)

# Estimate Button
if st.button("🔍 Estimate Cost"):
    adjusted_location_factor = regional_cost_index[selected_region] * location_factor
    estimated_cost_factor = (project_size * 200 * adjusted_location_factor) + (num_floors * 5000) + (timeline * 3000)

    # Ensure correct shape for model prediction
    input_data = np.array([[estimated_cost_factor] * scaler.n_features_in_])
    predicted_total_cost = model.predict(scaler.transform(input_data))[0]

    # Cost Breakdown
    material_cost = predicted_total_cost * 0.5  
    labor_cost = predicted_total_cost * 0.3  
    equipment_cost = predicted_total_cost * 0.1  
    additional_costs = predicted_total_cost * 0.1  

    # **Summary Table & Pie Chart Side-by-Side**
    col1, col2 = st.columns([1, 1])  

    with col1:
        st.write("📊 **Summary Cost Estimate**")
        summary_data = pd.DataFrame({
            "Category": ["Total Estimated Cost", "Material Cost", "Labor Cost", "Equipment Cost", "Additional Costs"],
            "Amount (USD)": [predicted_total_cost, material_cost, labor_cost, equipment_cost, additional_costs]
        })
        st.dataframe(summary_data)

        # Download Summary Table
        summary_excel = summary_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary Table", data=summary_excel, file_name="Summary_Cost_Estimate.csv", mime="text/csv")

    with col2:
        fig, ax = plt.subplots()
        ax.pie([material_cost, labor_cost, equipment_cost, additional_costs], labels=["Materials", "Labor", "Equipment", "Additional"], autopct='%1.1f%%')
        ax.set_title("Cost Breakdown")
        st.pyplot(fig)

    # **Detailed Cost Breakdown Matrix**
    st.write("📊 **Detailed Cost Breakdown**")
    cost_breakdown_matrix = pd.DataFrame({
        "Category": ["Materials", "Materials", "Materials", "Materials", "Labor", "Labor", "Labor", "Equipment", "Equipment", "Additional"],
        "Item": ["Concrete", "Steel", "Bricks", "Glass", "Skilled Workers", "Unskilled Workers", "Subcontractors", "Machinery", "Power Tools", "Permits"],
        "Unit Type": ["Cubic Meter", "Tons", "Pieces", "Sheets", "Hours", "Hours", "Contract", "Days", "Pieces", "Lump Sum"],
        "Quantity": [500, 100, 2000, 500, 8000, 5000, 10, 30, 100, 1],
        "Unit Price (USD)": [50, 700, 2, 15, 20, 15, 5000, 1000, 200, 5000],
        "Total Cost (USD)": [
            material_cost * 0.4, material_cost * 0.3, material_cost * 0.2, material_cost * 0.1,
            labor_cost * 0.5, labor_cost * 0.3, labor_cost * 0.2,
            equipment_cost * 0.7, equipment_cost * 0.3,
            additional_costs
        ]
    })
    st.dataframe(cost_breakdown_matrix)

    # Download Cost Breakdown Table
    cost_excel = cost_breakdown_matrix.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Detailed Cost Breakdown", data=cost_excel, file_name="Detailed_Cost_Breakdown.csv", mime="text/csv")

    # **🏗️ Cost Benchmarking Graph**
    st.write("📊 **Cost Benchmarking vs. Industry Standard**")
    industry_averages = {"Residential": 2000, "Industrial": 1500, "Commercial": 2500, "Infrastructure": 3000, "Education": 2200, "Healthcare": 2800}
    industry_cost = industry_averages.get(project_type, 2000) * project_size
    fig, ax = plt.subplots()
    ax.bar(["Estimated Cost", "Industry Avg"], [predicted_total_cost, industry_cost], color=["blue", "gray"])
    ax.set_ylabel("Total Cost (USD)")
    st.pyplot(fig)

# Footer
st.markdown("🚀 **AI-Powered Cost Estimator for Smart Budgeting!**")
