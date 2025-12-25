# Final Project for Programing For Data Science (CSC17104)
---
## 1. Project Overview & Team Info

Đồ án này tập trung giải quyết bài toán dự báo mưa tại Úc – một lục địa có mức độ đa dạng khí hậu rất cao, nơi các quy luật khí tượng mang tính tổng quát thường không đủ khả năng mô tả chính xác các biến động địa phương. Trước bối cảnh đó, đồ án được triển khai theo một quy trình khoa học dữ liệu có cấu trúc rõ ràng, gồm năm giai đoạn liên tiếp: thu thập dữ liệu, khám phá dữ liệu, xác định vấn đề, phân tích và diễn giải kết quả.

Thay vì chỉ dừng lại ở các thống kê mô tả hoặc tối ưu hóa mô hình thuần túy, đồ án áp dụng chiến lược phân tích đa tầng, trong đó trọng tâm là kiểm định các giả thuyết dựa trên cơ chế vật lý. Cách tiếp cận này cho phép kết hợp chặt chẽ kiến thức miền về khí tượng – thủy văn với các kỹ thuật phân tích dữ liệu hiện đại, nhằm đảm bảo rằng các kết luận rút ra không chỉ có ý nghĩa thống kê mà còn phù hợp với bản chất vật lý của các quá trình khí quyển.

Lộ trình phân tích của dự án được định hướng bởi sáu câu hỏi nghiên cứu (RQ1 – RQ6), được tổ chức thành hai trụ cột chính. Trụ cột thứ nhất tập trung vào việc làm rõ các cơ chế vật lý chi phối quá trình hình thành mưa, bao gồm vai trò của động lực học áp suất (RQ1), ảnh hưởng của tính mùa vụ (RQ2), tương tác giữa trạng thái nhiệt và độ ẩm (RQ3), cũng như các tín hiệu động lực học gió (RQ4). Trụ cột thứ hai hướng đến tối ưu hóa khía cạnh kỹ thuật dữ liệu, cụ thể là chiến lược xử lý dữ liệu thiếu (RQ5) và thiết kế, đánh giá các mô hình học máy trong bối cảnh dữ liệu mất cân bằng nghiêm trọng (RQ6).

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

1. **RQ1 (Pressure Dynamics):** Liệu mối quan hệ giữa sụt giảm áp suất khí quyển trong ngày (Pressure9am − Pressure3pm) và xác suất mưa ngày kế tiếp (`RainTomorrow`) có ổn định trên toàn bộ lãnh thổ Úc, hay hiệu lực dự báo của áp suất phụ thuộc phi tuyến vào trạng thái độ ẩm khí quyển và bối cảnh địa lý (Ven biển so với Nội địa)?

2. **RQ2 (Seasonality & Indicators):** Sự biến đổi của các đặc trưng khí tượng theo chu kỳ mùa vụ có mối tương quan thế nào với khả năng xảy ra hiện tượng mưa ngày tiếp theo, và đâu là những 'dấu hiệu' nhận biết sớm (early indicators) quan trọng nhất?

3. **RQ3 (Thermodynamic Interaction):** Phân tích sự tương tác phi tuyến giữa Biên độ nhiệt ngày (Diurnal Temperature Range - $DTR$) và Thời lượng nắng (Sunshine): Liệu sự kết hợp của hai yếu tố này có tạo ra một trạng thái 'Bất ổn định nhiệt ẩm' giúp dự báo mưa chính xác hơn các chỉ số đơn lẻ, và sự tương tác này thay đổi thế nào qua các nhóm trạm quan trắc khác nhau?

4. **RQ4 (Wind Dynamics):** Liệu chỉ số tương tác động lực gió, được xây dựng từ sự kết hợp giữa tốc độ gió giật (`WindGustSpeed`) và độ lệch hướng gió trong ngày (`Diurnal Wind Shift`), có thể đóng vai trò như một chỉ báo dự báo mưa ngày kế tiếp (`RainTomorrow`); hay hiệu lực dự báo của tín hiệu động lực này bị điều biến bởi trạng thái nhiệt – ẩm của khí quyển (đặc trưng bởi độ ẩm và nhiệt độ buổi chiều), cũng như bối cảnh động lực học khu vực (hệ thống gió ven biển so với đối lưu nội địa)?

5. **RQ5 (Data Preprocessing):** Phương pháp xử lý missing values nào (Simple Univariate Imputation vs. Multivariate Iterative Imputation) là tối ưu nhất để bảo toàn cấu trúc phân phối thống kê của các đặc trưng có tỷ lệ missing values cao trong dữ liệu (`Sunshine`, `Evaporation`, `Cloud9am`, `Cloud3pm`)?

6. **RQ6 (Predictive Modeling):** Trong bối cảnh dữ liệu mất cân bằng (tỷ lệ ngày mưa thấp hơn nhiều so với ngày không mưa), mô hình học máy nào thuộc ba nhóm kiến trúc chính: **Linear** (Logistic Regression), **Bagging** (Random Forest) hay **Boosting** (XGBoost) mang lại hiệu quả cao nhất trong việc tối đa hóa **Recall** (Độ nhạy) để giảm thiểu rủi ro bỏ sót dự báo mưa, trong khi vẫn duy trì chỉ số **F1-Score** ở mức chấp nhận được?

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
