  Folder source code gồm 2 folder chính:
__input__: folder chứa dữ liệu của bài toán. Bao gồm:
+ Dữ liệu gốc của bài toán: "jpx-tokyo-stock-exchange-prediction"
  + Sau khi Data Preprocessing: "1_DataPreprocessing"
  + Sau khi Feature Engineering: “2_FeatureEngineering”
  + Sau khi Feature Selection: “3_FeatureSelection”
+ working: Gồm các file *.ipynb - file chạy code; data_utils.py - chứa các hàm cơ bản; *.csv - các kết quả lưu ra. Trong đó, với file .ipynb:
  + Chia làm 5 phần: 
    + 0 - Phân tích dữ liệu ban đầu
    + 1 - Data Preprocessing
    + 2 - Feature Engineering
    + 3 - Feature Selection
    + 4 - ARIMA
  + Trong đó, với mục 1, 2, 3 là thực hiện từng bước xử lý dữ liệu cho mô hình XGBoost, gồm: xử lý dữ liệu, phân tích, chạy model.
  + Ngoài ra còn có file data_utils.py chứa một số hàm quan trọng:
    + train_valid_test_split: tách dữ liệu thành các tập train/valid/test
    + render_col: plot dữ liệu của 1 mã code theo số lượng ngày nhất định
    + dataPreprocessing_for_HiddenTest: tiền xử lý dữ liệu cho tập test ẩn
    + featureEngineering_for_HiddenTest: xử lý các thuộc tính cho tập test ẩn
    + calc_score: tính toán điểm của dự đoán so với thực tế
