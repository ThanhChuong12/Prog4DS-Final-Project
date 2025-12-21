# Prog4DS-Final-Project
Final Project for CSC17104 (PROGRAMING FOR DATA SCIENCE)
## 1. Project Overview & Team Info
Dự án thực hiện phân tích chuyên sâu trên bộ dữ liệu thời tiết của Úc nhằm giải mã các quy luật khí tượng phức tạp. Thay vì chỉ dừng lại ở các thống kê mô tả, dự án tập trung vào việc tìm kiếm các "ngòi nổ" vật lý (như sự sụt giảm áp suất, biên độ nhiệt hẹp) và so sánh hiệu quả của các kỹ thuật xử lý dữ liệu tiên tiến để tối ưu hóa khả năng dự báo mưa.

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
Dự án giải quyết 5 bài toán phân tích cốt lõi:
1. **Áp suất & Địa lý:** Phân tích hiện tượng "áp thấp giả" ở nội địa và sự khác biệt về cơ chế gây mưa giữa vùng ven biển và nội địa.
2. **Dấu hiệu sớm:** Xác định các biến số có tương quan mạnh nhất (Humidity, Sunshine) để làm tín hiệu dự báo.
3. **Tương tác Phi tuyến:** Nghiên cứu sự kết hợp giữa Biên độ nhiệt ($DTR$) và Nắng ($Sunshine$) để tìm trạng thái bất ổn định nhiệt động lực học.
4. **Mô hình hóa:** Xây dựng công thức và đánh giá khả năng dự báo dựa trên các biến số đã tinh lọc.
5. **Kỹ thuật Imputation:** So sánh hiệu quả giữa *Simple Imputer* và *Iterative Imputer (MICE)* trong việc giữ gìn logic vật lý của dữ liệu.

## 4. Key Findings Summary
*(wait)*
- **Insight 1:** ...
- **Insight 2:** ...

## 5. File Structure Explanation
Dự án được tổ chức theo cấu trúc Modular giúp dễ dàng bảo trì và tái sử dụng code:

- `data/raw/`: Chứa bộ dữ liệu gốc từ Kaggle (`weatherAUS.csv`).
- `data/processed/`: Chứa các tệp dữ liệu đã qua xử lý (`data.csv`) và các tập đã split (`train.csv`, `test.csv`).
- `notebooks/`: Các bước phân tích chi tiết:
    - `01_data_collection.ipynb`: Nạp và tổng quan dữ liệu ban đầu.
    - `02_data_exploration.ipynb`: Khám phá phân phối, xu hướng và dữ liệu thiếu (EDA).
    - `03_preprocessing.ipynb`: Pipeline làm sạch, mã hóa, MICE imputation và PCA.
    - `Question1.ipynb` -> `Question5.ipynb`: Các bài toán nghiên cứu chuyên sâu theo từng chủ đề.
- `src/`: Các module tái sử dụng:
    - `data_processing.py`: Các hàm Load, Clean, Label Encode, PCA, và Split dữ liệu.
    - `visualization.py`: Các hàm vẽ biểu đồ chuyên sâu (Missingness matrix, Correlation heatmap, Conditional probability).
    - `__init__.py`: Quản lý việc export các hàm trong package.
- `assets/images/`: Lưu trữ các hình ảnh để minh họa trong notebook và biểu đồ xuất ra từ Notebooks (KDE plots, Heatmaps, Boxplots).
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
Dự án được xây dựng trên các thư viện mã nguồn mở:
- pandas: Thao tác dữ liệu bảng.
- numpy: Tính toán ma trận và đại số tuyến tính.
- seaborn: Trực quan hóa dữ liệu thống kê cao cấp.
- matplotlib: Vẽ biểu đồ cơ bản.
- warnings: Quản lý các thông báo lỗi phiên bản hệ thống.
