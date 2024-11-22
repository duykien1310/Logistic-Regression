import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc tập dữ liệu (giả định bạn có file CSV 'loan_data.csv')
# Chỉnh sửa đường dẫn và cột tùy thuộc vào dữ liệu của bạn
data = pd.read_csv('train.csv')

# Giả định tập dữ liệu có các cột features và một cột 'target' (1: cho vay, 0: không cho vay)
features = data.drop(columns=['loan_status'])
target = data['loan_status']

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (nếu cần thiết)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện Logistic Regression với regularization L2
model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Nếu muốn kiểm tra kết quả với một đối tượng mới (ví dụ)
new_data = np.array([[...]])  # Thay thế bằng dữ liệu của đối tượng cần kiểm tra
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"Dự đoán: {'Cho vay' if prediction[0] == 1 else 'Không cho vay'}")
