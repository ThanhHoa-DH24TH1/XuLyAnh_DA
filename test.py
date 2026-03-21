import cv2
import os
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
# Chú ý: Thay bằng đúng thư mục bạn vừa gán nhãn
FOLDER_PATH = "D:\\XLA\\Dog-Breeds-Dataset\\akita dog" 

def test_yolo_labels():
    classes_file = os.path.join(FOLDER_PATH, "classes.txt")
    if not os.path.exists(classes_file):
        print("Lỗi: Không tìm thấy file classes.txt! Bạn đã gán nhãn chuẩn YOLO chưa?")
        return
        
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # --- ĐIỂM KHÁC BIỆT ---
    # Chỉ tìm những file .txt bạn đã tạo (bỏ qua file classes.txt)
    txt_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt') and f != 'classes.txt']
    
    if not txt_files:
        print("Chưa tìm thấy file nhãn (.txt) nào trong thư mục này.")
        return

    # Tính toán lưới hiển thị linh hoạt (hiển thị tất cả ảnh đã làm)
    n_images = len(txt_files)
    cols = 3
    rows = (n_images + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))

    print(f"Đã tìm thấy {n_images} ảnh được gán nhãn. Đang vẽ khung...")

    for i, txt_name in enumerate(txt_files):
        # Lấy tên gốc để ghép với đuôi ảnh
        base_name = os.path.splitext(txt_name)[0]
        
        # Tìm file ảnh tương ứng (xử lý cả đuôi .jpg, .jpeg, .png)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            temp_path = os.path.join(FOLDER_PATH, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            continue

        txt_path = os.path.join(FOLDER_PATH, txt_name)
        
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape 
        
        # Đọc tọa độ từ file .txt và vẽ
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.strip().split()
            if len(data) >= 5:
                class_id = int(data[0])
                x_center, y_center, width, height = map(float, data[1:5])
                
                # Giải mã tọa độ YOLO sang Pixel
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                # Vẽ khung màu đỏ
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # Ghi chữ
                if class_id < len(class_names):
                    label = class_names[class_id]
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
# Hiển thị lên lưới
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(base_name)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_yolo_labels()