Folder source code gồm 2 folder chính:
+ input: gồm data, bao gồm: 
  + data gốc của bài toán: "jpx-tokyo-stock-exchange-prediction"
  + data sau Data Preprocessing: "1_after_datapreprocessing"
  + data sau Feature Engineering: "2_after_featureEngineering"
  + data sau Feature Selection: "3_ after_feature_selection"
+ working: Gồm các file *.ipynb - file chạy code; data_utils.py - chứa các hàm cơ bản; *.csv - các kết quả lưu ra. Trong đó, với file .ipynb:
  + Chia làm 4 phần: 0 - ban đầu; 1 - Data Preprocessing; 2 - Feature Engineering; 3 - Feature Selection
  + Mỗi phần gồm 3 mục con: 0 - Xử lý dữ liệu; 1 - Phân tích dữ liệu; 2 - chạy mô hình