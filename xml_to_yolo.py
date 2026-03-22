import os
import xml.etree.ElementTree as ET

# === CẤU HÌNH ĐƯỜNG DẪN ===
# 1. Thư mục mẹ chứa các thư mục con Annotation (VD: D:\XLA\Dataset\Annotation)
ANNOTATION_DIR = "D:\\Downloads\\archive\\annotations\\Annotation" 

# 2. Thư mục mẹ chứa các thư mục con Hình ảnh (VD: D:\XLA\Dataset\Images)
IMAGES_DIR = "D:\\Downloads\\archive\\images\\Images" 

# Danh sách tự động lưu các giống chó
classes = []

def convert_to_yolo_bbox(size, box):
    """Công thức chuyển đổi tọa độ Pixel sang hệ YOLO (0.0 - 1.0)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    return (x_center * dw, y_center * dh, width * dw, height * dh)

def convert_stanford_dogs():
    print("Bắt đầu quét thư mục và chuyển đổi...")
    
    # Duyệt qua từng thư mục giống chó (VD: n02085620-Chihuahua)
    for breed_folder in os.listdir(ANNOTATION_DIR):
        ann_breed_path = os.path.join(ANNOTATION_DIR, breed_folder)
        img_breed_path = os.path.join(IMAGES_DIR, breed_folder)
        
        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(ann_breed_path):
            continue
            
        # Duyệt qua từng file tọa độ trong thư mục giống chó
        for ann_file in os.listdir(ann_breed_path):
            ann_filepath = os.path.join(ann_breed_path, ann_file)
            
            try:
                # Đọc file (dù không có đuôi .xml)
                tree = ET.parse(ann_filepath)
                root = tree.getroot()
                
                # Lấy kích thước ảnh
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                
                # Tạo tên file .txt tương ứng để lưu vào thư mục Images
                txt_filename = ann_file + ".txt"
                txt_filepath = os.path.join(img_breed_path, txt_filename)
                
                # Mở file txt để ghi
                with open(txt_filepath, 'w') as out_file:
                    for obj in root.iter('object'):
                        cls_name = obj.find('name').text
                        
                        # Tự động thêm tên chó vào danh sách nếu chưa có
                        if cls_name not in classes:
                            classes.append(cls_name)
                            
                        cls_id = classes.index(cls_name)
                        xmlbox = obj.find('bndbox')
                        
                        # Lấy tọa độ bndbox
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        
                        # Chuyển đổi và ghi vào file
                        bb = convert_to_yolo_bbox((w, h), b)
                        out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
                        
            except Exception as e:
                print(f"⚠️ Có lỗi ở file {ann_filepath}: {e}")
                
    # Xuất danh sách các loài chó ra file classes.txt
    classes_file_path = os.path.join(IMAGES_DIR, "classes.txt")
    with open(classes_file_path, 'w') as f:
        for cls in classes:
            f.write(cls + '\n')
            
    print(f"\n✅ HOÀN TẤT! Đã nhận diện được tổng cộng {len(classes)} giống chó.")
    print(f"Các file .txt đã được thả vào chung với thư mục chứa ảnh.")
    print(f"Danh sách loài chó được lưu tại: {classes_file_path}")

if __name__ == "__main__":
    convert_stanford_dogs()             