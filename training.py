from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
import pandas as pd
import mlflow

mlflow.set_experiment("Titanic Random Rorest")
mlflow.set_tracking_uri('http://127.0.0.1:5000')
# Load dữ liệu

data = pd.read_csv('./data/processed_data.csv')
X = data.drop(['Survived'], axis=1)
y = data['Survived']

# Xử lý mất cân bằng dữ liệu
survived_counts = y.value_counts()
print("\n Dữ liệu gốc: ")
print(survived_counts)

# Oversampling với SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
df_smote = pd.concat([X_smote, y_smote], axis=1)
df_smote.to_csv('./data/smote_data.csv')
print("\nPhân bố sau khi áp dụng SMOTE:")
print(pd.Series(y_smote).value_counts())

# Undersampling với NearMiss
nearmiss = NearMiss()
X_nm, y_nm = nearmiss.fit_resample(X, y)
df_nearmiss = pd.concat([X_nm, y_nm ], axis=1)
df_nearmiss.to_csv('./data/nearmiss_data.csv')
print("\nPhân bố sau khi áp dụng NearMiss:")
print(pd.Series(y_nm).value_counts())

# Lưu kết quả value_counts vào một dictionary để tạo DataFrame
value_counts_results = {
    'Method': ['Original', 'SMOTE', 'NearMiss'],
    'Class 0 Count': [survived_counts[0], pd.Series(y_smote).value_counts()[0], pd.Series(y_nm).value_counts()[0]],
    'Class 1 Count': [survived_counts[1], pd.Series(y_smote).value_counts()[1], pd.Series(y_nm).value_counts()[1]]
}

# Chuyển thành DataFrame
df_value_counts = pd.DataFrame(value_counts_results)

# Lưu kết quả vào file CSV
df_value_counts.to_csv('./data/value_counts_results.csv', index=False)



# Định nghĩa các tham số cần tối ưu hóa
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [None, 10, 20],      
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],     
    'max_features': ['sqrt', 'log2', None] 
}

# Các phương pháp xử lý mất cân bằng dữ liệu
methods = {
  "Original": [X, y],
  "SMOTE": [X_smote, y_smote],
  "NearMiss": [X_nm, y_nm]
}

# Danh sách để lưu kết quả mô hình
results = []

# Lặp qua các phương pháp
for dataset_name, (X_resampled, y_resampled) in methods.items():
    run_name = f"Random Forest_{dataset_name}" 
    with mlflow.start_run(run_name = run_name):
        print(f"\n{dataset_name}")
        
        # Chia dữ liệu thành train, validation và test cho mỗi phương pháp
        X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, shuffle=True)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
        mlflow.log_param('Train_size', '70%')
        mlflow.log_param('Test_size', '15%')
        mlflow.log_param('Val_size', '15%')

        # Khởi tạo mô hình Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # Đánh giá mô hình trên tập validation
        y_valid_pred = rf_model.predict(X_valid)
        valid_report = classification_report(y_valid, y_valid_pred, output_dict=True)

        # Đánh giá mô hình trên tập test
        y_test_pred = rf_model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)

        # Lưu kết quả vào danh sách
        results.append({
            'Method': dataset_name,
            'Validation Accuracy': valid_report['accuracy'],
            'Validation Precision': valid_report['weighted avg']['precision'],
            'Validation Recall': valid_report['weighted avg']['recall'],
            'Validation F1-Score': valid_report['weighted avg']['f1-score'],
            'Test Accuracy': test_report['accuracy'],
            'Test Precision': test_report['weighted avg']['precision'],
            'Test Recall': test_report['weighted avg']['recall'],
            'Test F1-Score': test_report['weighted avg']['f1-score']
        })
        mlflow.log_metric("Validation Accuracy", valid_report['accuracy'])
        mlflow.log_metric("Validation Precision", valid_report['weighted avg']['precision'])
        mlflow.log_metric("Validation Recall", valid_report['weighted avg']['recall'])
        mlflow.log_metric("Validation F1-Score", valid_report['weighted avg']['f1-score'])
        mlflow.log_metric("Test Accuracy", test_report['accuracy'])
        mlflow.log_metric("Test precision", test_report['weighted avg']['precision'])
        mlflow.log_metric("Test Recall", test_report['weighted avg']['recall'])
        mlflow.log_metric("Test F1-Score", test_report['weighted avg']['f1-score'])

        mlflow.sklearn.log_model(rf_model, run_name)

        if dataset_name == 'Orginal':
            mlflow.log_artifact('./data/data.csv')
        elif dataset_name == 'SMOTE':
            mlflow.log_artifact('./data/smote_data.csv')
        elif dataset_name == 'NearMiss':
            mlflow.log_artifact('./data/nearmiss_data.csv')

# Chuyển kết quả thành DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv('./data/model_results.csv', index=False)
