import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_yolo_prediction(image_path, model_path="best.pt"):
    """
    Hàm kiểm tra khả năng nhận diện của mô hình YOLO đã huấn luyện.
    
    :param image_path: Đường dẫn đến bức ảnh cần test (VD: ảnh chó Dogo Argentino)
    :param model_path: Đường dẫn đến file trọng số mô hình đã huấn luyện
    """
    try:
        # 1. Load mô hình YOLO đã được huấn luyện
        # Lưu ý: Cần đảm bảo file best.pt đang nằm cùng thư mục hoặc trỏ đúng đường dẫn
        model = YOLO(model_path)
        
        print(f"Đang tiến hành nhận diện ảnh: {image_path}...")
        
        # 2. Chạy dự đoán
        # Tham số conf=0.25 (Chỉ hiển thị các khung có độ tin cậy >= 25%)
        results = model.predict(source=image_path, conf=0.25)
        
        # 3. Lấy kết quả ảnh đã được vẽ sẵn bounding box, label và confidence từ YOLO
        annotated_img = results[0].plot()
        
        # Chuyển đổi hệ màu BGR của OpenCV sang RGB để hiển thị đúng màu bằng Matplotlib
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # 4. Hiển thị kết quả
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_img_rgb)
        plt.axis("off")
        plt.title("Kết quả nhận diện bằng YOLO")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        print("Vui lòng kiểm tra lại đường dẫn ảnh và file trọng số mô hình (best.pt)!")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ĐỂ TEST ---
    # Thay đường dẫn này bằng bức ảnh bạn muốn test. 
    # Ví dụ với ảnh bạn đã tải lên: "XuLyAnh_DA/dogo argentino/Image_1.jpg"
    TEST_IMAGE = "test.jpg" 
    
    # Đường dẫn file trọng số của bạn (cập nhật nếu bạn lưu ở nơi khác)
    WEIGHT_FILE = "runs/detect/train/weights/best.pt" 
    
    test_yolo_prediction(TEST_IMAGE, WEIGHT_FILE)