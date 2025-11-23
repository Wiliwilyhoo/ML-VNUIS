import joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# Load các mô hình đã lưu
model_dict = {
    "KNN": joblib.load("knn.pkl"),
    "XGBoost": joblib.load("XGBoost.pkl"),
    "Random Forest": joblib.load("RF.pkl"),
}
scaler = joblib.load("scaler.pkl")  

# Load danh sách feature đúng thứ tự
with open("feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

# Giao diện người dùng
st.title("💡 Dự đoán khách hàng có đăng ký gửi tiết kiệm")

model_name = st.selectbox("Chọn mô hình", list(model_dict.keys()))
model = model_dict[model_name]


# Khởi tạo dict với 0 cho tất cả cột

input_dict = {col: 0 for col in feature_names}

# Nhập một số cột số
input_dict['age'] = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
input_dict['balance'] = st.number_input("Số dư tài khoản", value=1000)
input_dict['campaign'] = st.number_input("Số lần liên hệ trong chiến dịch", value=1)
input_dict['pdays'] = st.number_input("Số ngày kể từ lần liên hệ trước", value=999)
input_dict['previous'] = st.number_input("Số lần liên hệ trước đó", value=0)

# Các biến nhị phân dạng yes/no (reverse encoding)
input_dict['loan'] = 0 if st.selectbox("Có vay nợ?", ['no', 'yes']) == 'no' else 1
input_dict['default'] = 0 if st.selectbox("Có nợ xấu?", ['no', 'yes']) == 'no' else 1
input_dict['housing'] = 0 if st.selectbox("Có vay nhà?", ['no', 'yes']) == 'no' else 1

# One-hot encoding: job
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
               'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
job = st.selectbox("Nghề nghiệp", job_options)
for j in job_options:
    input_dict[j] = 1 if job == j else 0

# One-hot encoding: education
edu_options = ['primary', 'secondary', 'tertiary']
education = st.selectbox("Trình độ học vấn", edu_options)
for edu in edu_options:
    input_dict[edu] = 1 if education == edu else 0

# One-hot encoding: marital
marital_options = ['married', 'single', 'divorced']
marital = st.selectbox("Tình trạng hôn nhân", marital_options)
for m in marital_options:
    input_dict[m] = 1 if marital == m else 0

# One-hot encoding: contact
contact_options = ['cellular', 'telephone']
contact = st.selectbox("Hình thức liên hệ", contact_options)
for c in contact_options:
    input_dict[c] = 1 if contact == c else 0

# One-hot encoding: month
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']
month = st.selectbox("Tháng liên hệ", month_options)
for m in month_options:
    input_dict[m] = 1 if month == m else 0

# One-hot encoding: poutcome
poutcome_options = ['failure', 'other', 'success']
poutcome = st.selectbox("Kết quả chiến dịch trước", poutcome_options)
for p in poutcome_options:
    input_dict[p] = 1 if poutcome == p else 0

# Dự đoán khi người dùng nhấn nút
if st.button("📊 Dự đoán"):
    # Chuẩn hóa dữ liệu đầu vào
    input_df = pd.DataFrame([input_dict])              
    input_df = input_df[feature_names]  
    # Scale
    input_scaled = scaler.transform(input_df)        
    if model_name == "LR":
        poly = PolynomialFeatures(degree=2)
        input_scaled = poly.fit_transform(input_scaled)
    # Dự đoán
    prediction = model.predict(input_scaled)[0]

    # Tính xác suất nếu model hỗ trợ
    proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    # Hiển thị kết quả
    if prediction == 1:
        st.success("✅ Khách hàng sẽ đăng ký gửi tiết kiệm!")
    else:
        st.warning("❌ Khách hàng sẽ không đăng ký.")

    if proba is not None:
        st.write(f"🎯 Xác suất đăng ký: **{proba:.2%}**")
