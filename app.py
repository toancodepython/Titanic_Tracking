import streamlit as st
import pandas as pd


st.set_page_config(
    page_title= "Titanic Random Forest"
)
# Load dữ liệu
raw_data_file = "../data/data.csv"  
processed_data_file = "../data/processed_data.csv"  
result_file = "../data/model_results.csv"  
smote_file = "../data/smote_data.csv"
nearmiss_file = "../data/nearmiss_data.csv"


df_raw = pd.read_csv(raw_data_file)
df_processed = pd.read_csv(processed_data_file)
df_results = pd.read_csv(result_file)
df_smote = pd.read_csv(smote_file)
df_nearmiss = pd.read_csv(nearmiss_file)

st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn phần hiển thị", ["Dữ liệu gốc", "Dữ liệu sau tiền xử lý", "Dữ liệu SMOTE", "Dữ liệu Nearmiss", "Kết quả mô hình"])

if page == "Dữ liệu gốc":
    st.title("Dữ liệu gốc")
    st.write(f"Số dòng: {df_raw.shape[0]} | Số cột: {df_raw.shape[1]}")
    st.dataframe(df_raw)
    if "Survived" in df_raw.columns:
        st.subheader("Phân bố lớp trong dữ liệu gốc:")
        class_counts = df_raw["Survived"].value_counts().reset_index()
        class_counts.columns = ["Lớp", "Số lượng"]
        st.table(class_counts)

elif page == "Dữ liệu SMOTE":
    st.title("Dữ liệu SMOTE")
    st.write(f"Số dòng: {df_smote.shape[0]} | Số cột: {df_smote.shape[1]}")
    st.dataframe(df_smote)
    if "Survived" in df_smote.columns:
        st.subheader(" Phân bố lớp trong dữ liệu gốc:")
        class_counts = df_smote["Survived"].value_counts().reset_index()
        class_counts.columns = ["Lớp", "Số lượng"]
        st.table(class_counts)

elif page == "Dữ liệu Nearmiss":
    st.title("Dữ liệu Nearmiss")
    st.write(f"Số dòng: {df_nearmiss.shape[0]} | Số cột: {df_nearmiss.shape[1]}")
    st.dataframe(df_nearmiss)
    if "Survived" in df_nearmiss.columns:
        st.subheader("Phân bố lớp trong dữ liệu gốc:")
        class_counts = df_nearmiss["Survived"].value_counts().reset_index()
        class_counts.columns = ["Lớp", "Số lượng"]
        st.table(class_counts)


elif page == "Dữ liệu sau tiền xử lý":
    st.title("Dữ liệu sau tiền xử lý")
    st.write(f"Số dòng: {df_processed.shape[0]} | Số cột: {df_processed.shape[1]}")
    st.dataframe(df_processed)
    if "Survived" in df_processed.columns:
        st.subheader("Phân bố lớp trong dữ liệu gốc:")
        class_counts = df_processed["Survived"].value_counts().reset_index()
        class_counts.columns = ["Lớp", "Số lượng"]
        st.table(class_counts)

elif page == "Kết quả mô hình":
    st.title("Kết quả huấn luyện mô hình Random Forest")
    method_selected = st.selectbox("🛠 Chọn phương pháp xử lý dữ liệu:", df_results["Method"].unique())

    # Lọc dữ liệu theo phương pháp
    df_filtered = df_results[df_results["Method"] == method_selected]

    # Hiển thị bảng kết quả
    st.subheader(f"Kết quả cho phương pháp: {method_selected}")
    st.dataframe(df_filtered)

    # Nút tải file kết quả
    st.download_button(
        label= "Tải file kết quả",
        data=df_results.to_csv(index=False),
        file_name="model_results.csv",
        mime="text/csv"
    )