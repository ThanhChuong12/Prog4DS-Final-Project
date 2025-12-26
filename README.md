# Australian Rain Prediction: From Dynamics to Modeling

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23EB4034.svg?style=for-the-badge&logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **Final Project for Programming For Data Science (CSC17104)**
>
> *Faculty of Information Technology, VNU-HCM University of Science*
---
## Table of Contents

- [Australian Rain Prediction: From Dynamics to Modeling](#australian-rain-prediction-from-dynamics-to-modeling)
  - [Table of Contents](#table-of-contents)
  - [1. About The Project](#1-about-the-project)
  - [2. Dataset Source \& Description](#2-dataset-source--description)
  - [3. Research Questions](#3-research-questions)
  - [4. Key Findings Summary](#4-key-findings-summary)
    - [1. Cơ chế vật lý](#1-cơ-chế-vật-lý)
    - [2. Kỹ thuật dữ liệu: MICE chiến thắng PCA](#2-kỹ-thuật-dữ-liệu-mice-chiến-thắng-pca)
    - [3. Hiệu năng mô hình: XGBoost và bài toán đánh đổi](#3-hiệu-năng-mô-hình-xgboost-và-bài-toán-đánh-đổi)
  - [5. Repository Structure](#5-repository-structure)
  - [6. Getting Started](#6-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
  - [7. Contributors](#7-contributors)
  - [8. License \& Acknowledgments](#8-license--acknowledgments)
    - [Academic Acknowledgments](#academic-acknowledgments)
    - [Data Attribution](#data-attribution)
    - [License](#license)

## 1. About The Project

Đồ án này tập trung giải quyết bài toán dự báo mưa tại Úc – một lục địa có mức độ đa dạng khí hậu rất cao, nơi các quy luật khí tượng mang tính tổng quát thường không đủ khả năng mô tả chính xác các biến động địa phương. Trước bối cảnh đó, đồ án được triển khai theo một quy trình khoa học dữ liệu có cấu trúc rõ ràng, gồm năm giai đoạn liên tiếp: 

`Data Collection` ➔ `Exploration` ➔ `Problem Definition` ➔ `Analysis` ➔ `Interpretation`.


Thay vì chỉ dừng lại ở các thống kê mô tả hoặc tối ưu hóa mô hình thuần túy, đồ án áp dụng chiến lược phân tích đa tầng, trong đó trọng tâm là kiểm định các giả thuyết dựa trên cơ chế vật lý. Cách tiếp cận này cho phép kết hợp chặt chẽ kiến thức miền về khí tượng – thủy văn với các kỹ thuật phân tích dữ liệu hiện đại, nhằm đảm bảo rằng các kết luận rút ra không chỉ có ý nghĩa thống kê mà còn phù hợp với bản chất vật lý của các quá trình khí quyển.

Lộ trình phân tích của dự án được định hướng bởi sáu câu hỏi nghiên cứu (RQ1 – RQ6), được tổ chức thành hai trụ cột chính. 

Trụ cột thứ nhất tập trung vào việc làm rõ các cơ chế vật lý chi phối quá trình hình thành mưa, bao gồm vai trò của động lực học áp suất (RQ1), ảnh hưởng của tính mùa vụ (RQ2), tương tác giữa trạng thái nhiệt và độ ẩm (RQ3), cũng như các tín hiệu động lực học gió (RQ4). Trụ cột thứ hai hướng đến tối ưu hóa khía cạnh kỹ thuật dữ liệu, cụ thể là chiến lược xử lý dữ liệu thiếu (RQ5) và thiết kế, đánh giá các mô hình học máy trong bối cảnh dữ liệu mất cân bằng nghiêm trọng (RQ6).

| **Trụ cột 1: Giải mã cơ chế vật lý** | **Trụ cột 2: Tối ưu hóa kỹ thuật dữ liệu** |
| :--- | :--- |
| Tập trung làm rõ các quy luật tự nhiên chi phối quá trình gây mưa. | Tập trung vào tính bền vững và hiệu năng của quy trình xử lý. |
| • **RQ1:** Động lực học áp suất (Pressure Dynamics)<br>• **RQ2:** Tính mùa vụ (Seasonality)<br>• **RQ3:** Tương tác nhiệt - ẩm (Thermodynamics)<br>• **RQ4:** Tín hiệu động lực học gió (Wind Dynamics) | • **RQ5:** Chiến lược xử lý dữ liệu thiếu (Imputation Strategy)<br>• **RQ6:** Đánh giá các mô hình học máy trên dữ liệu mất cân bằng (Imbalanced Modeling)|

## 2. Dataset Source & Description

Dữ liệu được sử dụng là *Rain in Australia*, tập hợp các quan sát khí tượng thực tế từ mạng lưới trạm đo của chính phủ Úc.

| Thông tin | Chi tiết |
| :--- | :--- |
| **Primary Source** | [Rain in Australia (Kaggle)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) |
| **Original Custodian** | [Australian Bureau of Meteorology (BOM)](http://www.bom.gov.au/climate/data) |
| **Temporal Coverage** | **10 năm** (01/11/2007 – 25/06/2017) |
| **Geospatial Coverage** | **49 trạm quan trắc** trải rộng khắp các vùng khí hậu chính (Nhiệt đới, Ôn đới, Sa mạc...) |
| **Observational Unit** | Daily Observations (Quan sát hằng ngày) |
| **Dataset Size** | $\approx$ 145,000 bản ghi với 23 biến số khí tượng. |

> **Description:** Bộ dữ liệu bao gồm các biến số khí quyển cốt lõi như nhiệt độ, độ ẩm, áp suất, gió (tốc độ & hướng), nắng và mây. Biến mục tiêu là `RainTomorrow` (Ngày mai có mưa hay không).

## 3. Research Questions
Dự án giải quyết 6 bài toán phân tích chính:

1. **RQ1 (Pressure Dynamics):** Liệu mối quan hệ giữa sụt giảm áp suất khí quyển trong ngày (Pressure9am − Pressure3pm) và xác suất mưa ngày kế tiếp (`RainTomorrow`) có ổn định trên toàn bộ lãnh thổ Úc, hay hiệu lực dự báo của áp suất phụ thuộc phi tuyến vào trạng thái độ ẩm khí quyển và bối cảnh địa lý (Ven biển so với Nội địa)?

2. **RQ2 (Seasonality & Indicators):** Sự biến đổi của các đặc trưng khí tượng theo chu kỳ mùa vụ có mối tương quan thế nào với khả năng xảy ra hiện tượng mưa ngày tiếp theo, và đâu là những 'dấu hiệu' nhận biết sớm (early indicators) quan trọng nhất?

3. **RQ3 (Thermodynamic Interaction):** Phân tích sự tương tác phi tuyến giữa Biên độ nhiệt ngày (Diurnal Temperature Range - $DTR$) và Thời lượng nắng (Sunshine): Liệu sự kết hợp của hai yếu tố này có tạo ra một trạng thái 'Bất ổn định nhiệt ẩm' giúp dự báo mưa chính xác hơn các chỉ số đơn lẻ, và sự tương tác này thay đổi thế nào qua các nhóm trạm quan trắc khác nhau?

4. **RQ4 (Wind Dynamics):** Liệu chỉ số tương tác động lực gió, được xây dựng từ sự kết hợp giữa tốc độ gió giật (`WindGustSpeed`) và độ lệch hướng gió trong ngày (`Diurnal Wind Shift`), có thể đóng vai trò như một chỉ báo dự báo mưa ngày kế tiếp (`RainTomorrow`); hay hiệu lực dự báo của tín hiệu động lực này bị điều biến bởi trạng thái nhiệt – ẩm của khí quyển (đặc trưng bởi độ ẩm và nhiệt độ buổi chiều), cũng như bối cảnh động lực học khu vực (hệ thống gió ven biển so với đối lưu nội địa)?

5. **RQ5 (Data Preprocessing):** Phương pháp xử lý missing values nào (Simple Univariate Imputation vs. Multivariate Iterative Imputation) là tối ưu nhất để bảo toàn cấu trúc phân phối thống kê của các đặc trưng có tỷ lệ missing values cao trong dữ liệu (`Sunshine`, `Evaporation`, `Cloud9am`, `Cloud3pm`)?

6. **RQ6 (Predictive Modeling):** Trong bối cảnh dữ liệu mất cân bằng (tỷ lệ ngày mưa thấp hơn nhiều so với ngày không mưa), mô hình học máy nào thuộc ba nhóm kiến trúc chính: **Linear** (Logistic Regression), **Bagging** (Random Forest) hay **Boosting** (XGBoost) mang lại hiệu quả cao nhất trong việc tối đa hóa **Recall** (Độ nhạy) để giảm thiểu rủi ro bỏ sót dự báo mưa, trong khi vẫn duy trì chỉ số **F1-Score** ở mức chấp nhận được?

## 4. Key Findings Summary
Dự án không chỉ dừng lại ở việc đạt độ chính xác cao mà còn cung cấp những hiểu biết sâu sắc về cơ chế gây mưa và tính hiệu quả của các kỹ thuật xử lý dữ liệu. Dưới đây là 3 phát hiện cốt lõi:

### 1. Cơ chế vật lý
Phân tích dữ liệu đã lượng hóa thành công các quy luật khí tượng:
* **Ngưỡng "Khóa" Độ ẩm:** Sụt giảm áp suất ($\Delta P$) chỉ là điều kiện cần. Mưa chỉ thực sự được kích hoạt khi **Độ ẩm 3pm vượt ngưỡng 60%**.
* **Hiệu ứng "Chăn mây":** Trạng thái Biên độ nhiệt hẹp (DTR < 7°C) kết hợp với lượng nắng thấp là chỉ báo mạnh nhất của mưa. Mây dày ngăn cản bức xạ nhiệt ban đêm, giữ cho chênh lệch nhiệt độ ngày-đêm ở mức thấp.
* **Áp thấp giả:** Tại vùng nội địa khô hạn, áp suất giảm mạnh thường chỉ báo hiệu các đợt nắng nóng khô chứ không gây mưa nếu thiếu sự hội tụ ẩm.

### 2. Kỹ thuật dữ liệu: MICE chiến thắng PCA
* **Chiến lược điền khuyết (Imputation):** Phương pháp *Iterative Imputer (MICE)* với `max_iter=20` đã chứng minh tính ưu việt vượt trội so với điền trung bình. Nó giúp khôi phục cấu trúc dữ liệu, nâng hệ số tương quan vật lý giữa các biến từ *0.59 lên 0.67*.
* **PCA Failure:** Việc áp dụng PCA làm sụt giảm nghiêm trọng chỉ số *Recall*. Điều này chứng minh rằng các biến gốc mang thông tin biên giới phi tuyến cực kỳ quan trọng cho việc phân loại, mà khi chiếu sang không gian mới đã bị mất mát.

### 3. Hiệu năng mô hình: XGBoost và bài toán đánh đổi
Đánh giá hiệu quả của mô hình tập trung chủ yếu vào Recall và F1-Score vì nó hướng đến giá trị thực tế là ưu tiên cho việc thà chấp nhận dự đoán sai có mưa nhưng thực tế lại nắng còn hơn là bỏ sót đi những ngày có mưa thực sự.

* **XGBoost** vượt qua Random Forest và Logistic Regression trên mọi chỉ số quan trọng:
    * **AUC-ROC:** **0.892** (Khả năng phân loại xuất sắc).
    * **Recall:** **0.78** (Bắt được 78% số ngày mưa thực tế).
* **Tối ưu hóa Ngưỡng:** Ở ngưỡng mặc định (0.5), Recall chỉ đạt $\approx$ 0.5. Chúng tôi đã điều chỉnh *Decision Threshold* xuống thấp hơn để tối đa hóa khả năng cảnh báo thiên tai, chấp nhận đánh đổi một lượng nhỏ Precision để đạt mức Recall an toàn cho ứng dụng thực tế.


## 5. Repository Structure
Dự án được tổ chức theo kiến trúc *Modular*, tách biệt rõ ràng giữa Dữ liệu (Data), Mã nguồn (Source Code) và Phân tích (Notebooks). Cấu trúc này đảm bảo tính tái lập và dễ dàng bảo trì.

```text
├── assets/
│   └── images/              # Lưu trữ biểu đồ xuất ra từ Notebook (Heatmaps, KDE, Boxplots)
│
├── data/
│   ├── raw/                 # Dữ liệu thô nguyên bản - `weatherAUS.csv`
│   └── processed/           # Dữ liệu sạch, Feature Engineering & Train/Test splits
│
├── notebooks/               # Các bước phân tích & Thí nghiệm (Jupyter Notebooks)
│   ├── 01_data_collection.ipynb    # Tổng quan & Kiểm định nguồn dữ liệu
│   ├── 02_data_exploration.ipynb   # EDA: Phân phối, Outliers & Missing Patterns
│   ├── 03_preprocessing.ipynb      # Main Pipeline: Clean, MICE Imputation, Scaling, Split
│   ├── 04_project_summary.ipynb    # Tổng kết dự án & Reflections
│   └── analysis_RQs/               # Các câu hỏi nghiên cứu chuyên sâu
│       ├── Question1.ipynb         # RQ1: Pressure & Humidity Dynamics
│       ├── ...
│       └── Question6.ipynb         # RQ6: Modeling & Optimization
│
├── src/                     # Source Code
│   ├── __init__.py          # Package initializer
│   ├── data_processing.py   # Module xử lý dữ liệu (Load, Clean, Encode, Split)
│   └── visualization.py     # Module vẽ biểu đồ chuẩn hóa (Custom plotting functions)
│
├── .gitignore               # Các file/folder bị bỏ qua (vd: venv, .DS_Store)
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn dự án
```

## 6. Getting Started

Để chạy được dự án này trên máy cục bộ (local machine), vui lòng làm theo các hướng dẫn chi tiết dưới đây. Nhóm khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột phiên bản thư viện.

### Prerequisites
* **Python**: Phiên bản 3.8 trở lên.
* **Package Manager**: `pip` hoặc `conda`.
* **Git**: Để clone repository.
* 
### Installation

**Bước 1: Clone Repository**

Mở terminal và chạy lệnh sau để tải mã nguồn về máy:
```bash
git clone [https://github.com/ThanhChuong12/Prog4DS-Final-Project.git](https://github.com/ThanhChuong12/Prog4DS-Final-Project.git)
cd Prog4DS-Final-Project
```
**Bước 2: Thiết lập môi trường ảo (Virtual Environment)** 

*Lựa chọn 1 trong 2 cách sau:*
* Cách A: Sử dụng venv (Khuyến nghị cho VS Code)
```bash
python -m venv venv
# Kích hoạt môi trường (Windows):
venv\Scripts\activate
# Kích hoạt môi trường (macOS/Linux):
source venv/bin/activate
```
* Cách B: Sử dụng conda (Khuyến nghị cho Jupyter Lab)
```bash
conda create --name weather-env python=3.9
conda activate weather-env
```
**Bước 3: Cài đặt thư viện phụ thuộc**

Sau khi kích hoạt môi trường, chạy lệnh sau để cài đặt toàn bộ các thư viện cần thiết:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**Bước 4: Thiết lập dữ liệu**
1. Tải bộ dữ liệu `weatherAUS.csv` từ [Kaggle Link](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
2. Đặt file vào thư mục: `data/raw/weatherAUS.csv`.

### Usage
Dự án được thiết kế để chạy theo trình tự logic. Vui lòng thực thi các Notebook theo thứ tự sau để đảm bảo luồng dữ liệu chính xác:

1. **Khởi động Jupyter:**
```bash
jupyter notebook

```
*(Hoặc mở thư mục dự án bằng VS Code và chọn Kernel là môi trường ảo vừa tạo).*

2. **Trình tự thực thi:**
* `01_data_collection.ipynb`: Kiểm tra và nạp dữ liệu.
* `02_data_exploration.ipynb`: Chạy EDA để hiểu dữ liệu.
* `03_preprocessing.ipynb`: *Bắt buộc chạy* để tạo ra file `data/processed/train.csv` và `test.csv`.
* `QuestionX.ipynb`: Sau khi có dữ liệu processed, bạn có thể chạy bất kỳ notebook phân tích chuyên sâu nào (RQ1 - RQ6).
> **Lưu ý:** Quá trình chạy `MICE Imputation` trong bước Preprocessing có thể mất từ 5 - 10 phút tùy thuộc vào cấu hình máy tính của bạn.

## 7. Contributors

Dự án được thực hiện bởi nhóm gồm 03 thành viên thuộc *Khoa Công nghệ Thông tin - Trường Đại học Khoa học Tự nhiên, ĐHQG - HCM*.

| Sinh viên | MSSV | Vai trò | Trách nhiệm chính |
| :--- | :---: | :--- | :--- |
| **Lê Hà Thanh Chương** | `23120195` | **Project Lead**<br>Data Architect | • **Architecture:** Thiết kế hệ thống và quản trị dữ liệu.<br>• **Analysis:** Giải mã động lực học gió & áp suất (RQ1, RQ4).<br>• **Reporting:** Biên soạn biện giải khoa học. <br>• **Docs:** Tài liệu hóa dự án và trực quan hóa. |
| **Võ Trần Duy Hoàng** | `23120266` | **ML Engineer**<br>Quality Control | • **Modeling:** Tối ưu hóa mô hình và tuning tham số.<br>• **Strategy:** Triển khai chiến lược Imputation (RQ5, RQ6).<br>• **QC:** Kiểm soát chất lượng mã nguồn. |
| **Trần Đình Thi** | `23120359` | **Data Engineer**<br>Tech Writer | • **Engineering:** Triển khai MICE và feature Engineering.<br>• **Analysis:** Phân tích nhiệt - ẩm và mùa vụ (RQ2, RQ3).<br> • **Docs:** Biên soạn tài liệu README.md và tổng hợp kết quả từ sáu câu hỏi nghiên cứu.|

## 8. License & Acknowledgments

### Academic Acknowledgments
Dự án này là sản phẩm báo cáo cuối kỳ cho môn học *Lập trình cho Khoa học Dữ liệu (Programming for Data Science - CSC17104)* tại *Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*.

Nhóm thực hiện xin gửi lời cảm ơn chân thành đến quý thầy trong đội ngũ giảng dạy đã tận tình hướng dẫn, cung cấp kiến thức nền tảng và hỗ trợ kỹ thuật trong suốt quá trình thực hiện đồ án:
* **Instructor:** Mr. Phạm Trọng Nghĩa
* **Instructor:** Mr. Lê Nhựt Nam
* **Instructor:** Mr. Nguyễn Thanh Tình

### Data Attribution
Bộ dữ liệu sử dụng trong đồ án được trích xuất từ *[Cục Khí tượng Úc (Bureau of Meteorology - BOM)](http://www.bom.gov.au/)* thông qua nền tảng Kaggle. Chúng tôi ghi nhận và trân trọng sự đóng góp của BOM trong việc cung cấp nguồn dữ liệu khí tượng chất lượng cao phục vụ cho mục đích nghiên cứu và giáo dục.

### License
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Mã nguồn của dự án được phân phối dưới *Giấy phép MIT (MIT License)*.
Điều này đồng nghĩa với việc bạn được quyền tự do sử dụng, sao chép, sửa đổi, hợp nhất, xuất bản, phân phối mã nguồn này, miễn là giữ nguyên thông báo bản quyền gốc. Xem file `LICENSE` để biết thêm chi tiết.

<br>
<p align="center">
  <i>Built with ❤️ by the Prog4DS Team | December 2025</i>
</p>