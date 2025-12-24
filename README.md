# Prog4DS-Final-Project
Final Project for CSC17104 (PROGRAMING FOR DATA SCIENCE)
## 1. Project Overview & Team Info
Dự án này tập trung giải quyết bài toán dự báo mưa tại Úc – một lục địa có khí hậu đa dạng và phức tạp. Thay vì chỉ sử dụng các mô hình học máy thuần túy, chúng tôi tiếp cận vấn đề thông qua việc hiểu rõ bản chất vật lý của các biến số khí tượng và tối ưu hóa quy trình xử lý dữ liệu để đạt hiệu suất dự báo cao nhất.

**Team Members:**
- [Lê Hà Thanh Chương] - MSSV: 23120195 
- [Võ Trần Duy Hoàng] - MSSV: 23120266 
- [Trần Đình Thi] - MSSV: 23120359 


## 2. Dataset Source & Description
* **Primary Source (Platform):** [Rain in Australia (Kaggle)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
* **Original Custodian:** [Australian Bureau of Meteorology (BOM)](http://www.bom.gov.au/climate/data)
* **Temporal Coverage:** 10 năm (01/11/2007 đến 25/06/2017).
* **Geospatial Coverage:** 49 địa điểm khác nhau trải rộng khắp các vùng khí hậu chính ở Úc.
* **Last Update:** 5 năm trước.
* **Description:** Dữ liệu quan sát khí tượng hằng ngày được ghi nhận tại mạng lưới trạm đo thời tiết trên khắp nước Úc trong giai đoạn 2007 – 2017.


## 3. Research Questions List
Dự án giải quyết 6 bài toán phân tích chính:

1. **Q1 (Seasonal Patterns):** Sự biến thiên của Nhiệt độ và Độ ẩm theo mùa ảnh hưởng thế nào đến xác suất mưa?

2. **Q2 (Early Indicators):** Đâu là những dấu hiệu sớm (threshold) của Độ ẩm và Áp suất báo hiệu mưa?

3. **Q3 (Interaction Analysis):** Sự kết hợp giữa Biên độ nhiệt (DTR) và Lượng nắng tạo ra các vùng xác suất mưa phi tuyến như thế nào?

4. **Q4 (Wind Dynamics):** Tốc độ gió giật và sự thay đổi hướng gió đóng vai trò gì trong các cơn dông đối lưu?

5. **Q5 (Imputation Strategy):** Kỹ thuật MICE cải thiện tính nhất quán vật lý của dữ liệu khuyết thiếu ra sao so với điền trung bình?

6. **Q6 (Modeling):** So sánh hiệu suất của Logistic Regression, Random Forest và XGBoost trong việc dự báo mưa.

## 4. Key Findings Summary

- **Tín hiệu vật lý:**
    - Ngưỡng độ ẩm: Mưa thường xảy ra khi Độ ẩm 3pm vượt ngưỡng 60%.

    - Hiệu ứng "Chăn mây": Khi DTR hẹp (<7°C) kết hợp với Lượng nắng thấp, xác suất mưa bùng nổ do mây dày giữ nhiệt ban đêm.

    - Áp thấp giả: Tại vùng nội địa Úc, áp suất giảm mạnh chưa chắc có mưa nếu thiếu độ ẩm hội tụ.

- **Tiền xử lý dữ liệu:** 
    - Chiến lược MICE: Việc sử dụng MICE (max_iter=20) giúp tăng hệ số tương quan vật lý từ 0.59 lên 0.67, khử nhiễu tốt hơn điền trung bình.

    - PCA Failure: Việc giảm chiều bằng PCA bị loại bỏ do làm sụt giảm nghiêm trọng chỉ số Recall, chứng tỏ các biến số gốc mang thông tin phân loại cực kỳ quan trọng.

- **Kết quả Mô hình:** 
    - XGBoost đạt hiệu suất tốt nhất với AUC-ROC: 0.892.

    - Thách thức lớn nhất là chỉ số Recall thấp (~0.5) ở ngưỡng mặc định, đòi hỏi phải điều chỉnh Decision Threshold để không bỏ lỡ các cảnh báo thiên tai.
## 5. File Structure Explanation
Dự án được tổ chức theo cấu trúc Modular giúp dễ dàng bảo trì và tái sử dụng code:

- `data/raw/`: Chứa bộ dữ liệu gốc từ Kaggle (`weatherAUS.csv`).
- `data/processed/`: Chứa các tệp dữ liệu đã qua xử lý (`data.csv`) và các tập đã split (`train.csv`, `test.csv`).
- `notebooks/`: Các bước phân tích chi tiết:
    - `01_data_collection.ipynb`: Nạp và tổng quan dữ liệu ban đầu.
    - `02_data_exploration.ipynb`: Khám phá phân phối, xu hướng và dữ liệu thiếu (EDA).
    - `03_preprocessing.ipynb`: Pipeline làm sạch, mã hóa, MICE imputation và PCA.
    - `04_project_summary.ipynb`: Tổng kết toàn bộ dự án và bài học kinh nghiệm (Reflections).
    - `Question1.ipynb` -> `Question6.ipynb`: Các bài toán nghiên cứu chuyên sâu theo từng chủ đề.
- `src/`: Các module tái sử dụng:
    - `data_processing.py`: Các hàm Load, Clean, Label Encode, PCA, và Split dữ liệu.
    - `visualization.py`: Các hàm vẽ biểu đồ chuyên sâu (Missingness matrix, Correlation heatmap, Conditional probability).
    - `__init__.py`: Quản lý việc export các hàm trong package.
- `assets/images/`: Lưu trữ các biểu đồ và hình ảnh xuất ra từ Notebooks (KDE plots, Heatmaps, Boxplots).
- `requirements.txt`: Danh sách các thư viện phụ thuộc (pandas, seaborn, scikit-learn, missingno, statsmodels).
- `README.md`: Tệp hướng dẫn này.

## 6. How to Run Instructions
Để chạy dự án này, hãy đảm bảo bạn đã cài đặt môi trường Python phù hợp:
1. Clone repository:  
   `git clone https://github.com/ThanhChuong12/Prog4DS-Final-Project.git`
2. Cài đặt môi trường: Đảm bảo bạn đã cài đặt đúng Python và Conda
3. Cài đặt thư viện:  
   ```pip install -r requirements.txt```
4. Chạy phân tích: Mở tệp Notebook trong VS Code hoặc Jupyter Lab và thực thi các cell từ trên xuống dưới.

## 7. Dependencies List
Dự án được xây dựng trên ngôn ngữ Python và các thư viện sau:
- pandas & numpy: Xử lý cấu trúc dữ liệu.
- matplotlib & seaborn: Trực quan hóa dữ liệu nâng cao.
- scikit-learn: Tiền xử lý (MICE imputer, Scaler) và mô hình hóa (Logistic, Random Forest).
- xgboost: Thuật toán Gradient Boosting hiệu suất cao.
- scipy & statsmodels: Phân tích thống kê và hồi quy.
