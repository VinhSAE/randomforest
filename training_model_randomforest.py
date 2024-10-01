from osgeo import gdal, ogr, gdal_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import seaborn as sn
import glob
import os
from joblib import dump

# Định nghĩa các thư mục chứa dữ liệu
image_train_folder = './demo/img'
image_val_folder = './demo/img'
shapefile_train_folder = './demo/label'
shapefile_val_folder = './demo/label'
results_txt = './demo/results_txt_img_demo_RF_small04_.txt'
model_path = './demo/random_forest_model.joblib'

# Hàm đọc ảnh từ tệp
def read_image(file_path):
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    img = np.zeros((ds.RasterYSize, ds.RasterXSize, ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = ds.GetRasterBand(b + 1).ReadAsArray()
    return img, ds.GetProjection(), ds.GetGeoTransform()

# Hàm đọc shapefile và chuyển đổi thành mảng nhãn
def read_shapefile(file_path, img, img_proj, img_geo_transform, attribute):
    shape_ds = ogr.Open(file_path)
    shape_layer = shape_ds.GetLayer()

    # Tạo raster in-memory để rasterize shapefile
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('', img.shape[1], img.shape[0], 1, gdal.GDT_UInt16)
    mem_raster.SetProjection(img_proj)
    mem_raster.SetGeoTransform(img_geo_transform)
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    att_ = 'ATTRIBUTE=' + attribute
    err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1], [att_, "ALL_TOUCHED=TRUE"])
    assert err == gdal.CE_None

    return mem_raster.ReadAsArray()

# Hàm kết hợp dữ liệu ảnh và shapefile
def process_image_and_labels(img_files, shapefile_files, attribute):
    X_list = []
    y_list = []

    for img_file in img_files:
        img, img_proj, img_geo_transform = read_image(img_file)
        base_name = os.path.basename(img_file).replace('.tif', '')

        # Tìm shapefile tương ứng
        shapefile_file = [s for s in shapefile_files if base_name in os.path.basename(s)]
        if not shapefile_file:
            print(f'No matching shapefile for {img_file}')
            continue

        shapefile_file = shapefile_file[0]
        print(f'Processing shapefile: {shapefile_file}')
        roi = read_shapefile(shapefile_file, img, img_proj, img_geo_transform, attribute)

        # Xử lý dữ liệu
        X = img[roi > 0, :]
        y = roi[roi > 0]

        # Thêm vào danh sách
        X_list.append(X)
        y_list.append(y)

    # Kết hợp tất cả dữ liệu
    X_combined = np.vstack(X_list)
    y_combined = np.concatenate(y_list)

    return X_combined, y_combined

# Hàm huấn luyện mô hình Random Forest
def train_rf_model(X_train, y_train, n_estimators=150, n_jobs=-1):
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, verbose=1, n_jobs=n_jobs)
    X_train = np.nan_to_num(X_train)
    rf.fit(X_train, y_train)
    return rf

# Hàm đánh giá mô hình
def evaluate_model(rf, X_val, y_val, results_txt):
    X_val = np.nan_to_num(X_val)
    y_pred = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)

    print(f'Validation Accuracy: {accuracy}')
    print(f'Kappa Coefficient: {kappa}')

    # Ghi kết quả vào file
    with open(results_txt, "a") as file:
        file.write(f'Validation Accuracy: {accuracy}\n')
        file.write(f'Kappa Coefficient: {kappa}\n')

    return y_pred

# Hàm lưu mô hình và ghi các chỉ số OOB, tầm quan trọng các băng thông
def save_model_and_metrics(rf, X_train, model_path, results_txt):
    dump(rf, model_path)
    print(f'Model saved to {model_path}', file=open(results_txt, "a"))

    with open(results_txt, "a") as file:
        file.write('--------------------------------\n')
        file.write('TRAINING and RF Model Diagnostics:\n')
        file.write(f'OOB prediction of accuracy is: {rf.oob_score_ * 100:.2f}%\n')

        bands = range(1, X_train.shape[1] + 1)
        for b, imp in zip(bands, rf.feature_importances_):
            file.write(f'Band {b} importance: {imp:.4f}\n')

# Hàm tạo confusion matrix
def create_confusion_matrix(y_val, y_pred, results_txt, cm_path):
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('classes - predicted')
    plt.ylabel('classes - truth')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path)

    # Ghi confusion matrix vào file
    df = pd.DataFrame()
    df['truth'] = y_val
    df['predict'] = y_pred

    with open(results_txt, "a") as file:
        file.write(pd.crosstab(df['truth'], df['predict'], margins=True).to_string())
        file.write('\n')

# Main workflow
def main():
    # Đọc tất cả ảnh huấn luyện và nhãn
    train_image_files = glob.glob(os.path.join(image_train_folder, '*.tif'))
    train_shapefile_files = glob.glob(os.path.join(shapefile_train_folder, '*.shp'))
    X_train, y_train = process_image_and_labels(train_image_files, train_shapefile_files, 'class')

    # Đọc tất cả ảnh kiểm tra và nhãn
    val_image_files = glob.glob(os.path.join(image_val_folder, '*.tif'))
    val_shapefile_files = glob.glob(os.path.join(shapefile_val_folder, '*.shp'))
    X_val, y_val = process_image_and_labels(val_image_files, val_shapefile_files, 'class')

    # Huấn luyện mô hình Random Forest
    rf = train_rf_model(X_train, y_train)

    # Đánh giá mô hình trên tập huấn luyện
    print("Evaluating model on training set...")
    y_train_pred = evaluate_model(rf, X_train, y_train, results_txt)

    # Đánh giá mô hình trên tập kiểm tra
    print("Evaluating model on validation set...")
    y_val_pred = evaluate_model(rf, X_val, y_val, results_txt)

    # Lưu mô hình và ghi các chỉ số
    save_model_and_metrics(rf, X_train, model_path, results_txt)

    # Tạo confusion matrix và ghi vào file
    create_confusion_matrix(y_val, y_val_pred, results_txt, './demo/validation_confusion_matrix.png')
    create_confusion_matrix(y_train, y_train_pred, results_txt, './demo/training_confusion_matrix.png')

if __name__ == "__main__":
    main()