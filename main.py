import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Đọc tập dữ liệu
data = pd.read_csv('train.csv')

# Xử lý dữ liệu phân loại (label encoding cho các cột phân loại)
categorical_columns = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Xử lý dữ liệu thiếu (nếu có)
data = data.fillna(0)  # Thay thế giá trị thiếu bằng 0 (có thể thay đổi tùy thuộc vào logic của bạn)

# Tách features và target
features = data.drop(columns=['loan_status'])
target = data['loan_status']

# Chuẩn hóa dữ liệu (Standard Scaling cho các giá trị số)
scaler = StandardScaler()
X_train = scaler.fit_transform(features)

# Huấn luyện Logistic Regression với regularization L2
model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
model.fit(X_train, target)

# Đọc file test.csv
test_data = pd.read_csv('test.csv')

# Xử lý dữ liệu phân loại trên file test.csv giống như train.csv
for col in categorical_columns:
    test_data[col] = LabelEncoder().fit_transform(test_data[col])

# Xử lý dữ liệu thiếu (nếu có) trên test.csv
test_data = test_data.fillna(0)

# Tách features (bỏ cột 'loan_status') và target (nếu muốn so sánh)
X_test_data = test_data.drop(columns=['loan_status'])
y_test_data = test_data['loan_status']

# Chuẩn hóa dữ liệu test.csv
X_test_data_scaled = scaler.transform(X_test_data)

# Dự đoán trên dữ liệu test.csv
y_test_pred = model.predict(X_test_data_scaled)

# Đánh giá kết quả
test_accuracy = accuracy_score(y_test_data, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

print("\nClassification Report (Test Data):")
print(classification_report(y_test_data, y_test_pred))

print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test_data, y_test_pred))

# In kết quả dự đoán cho từng hàng trong test.csv
test_data['Prediction'] = y_test_pred
test_data['Prediction_Label'] = test_data['Prediction'].apply(lambda x: 'Cho vay' if x == 1 else 'Không cho vay')
print("\nKết quả dự đoán trên file test.csv:")
print(test_data[['person_age', 'loan_status', 'Prediction_Label']])
