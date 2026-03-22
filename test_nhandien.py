from ultralytics import YOLO
import cv2

# 1. Gọi cái "não bộ" bro vừa train về
print("Đang tải mô hình AI...")
model = YOLO('best.pt')

# 2. Đưa ảnh vào cho nó nhận diện (nhớ đổi tên ảnh cho đúng file của bro)
image_path = 'test_cho.jpg'
results = model(image_path)

# 3. Hiển thị kết quả lên màn hình
for result in results:
    # Lấy ảnh đã được AI vẽ khung và dán nhãn
    annotated_frame = result.plot()
    
    # Hiển thị cái ảnh đó ra một cửa sổ mới
    cv2.imshow("Ket qua nhan dien AI - Do an cua Duc", annotated_frame)
    
    # Nhấn phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()