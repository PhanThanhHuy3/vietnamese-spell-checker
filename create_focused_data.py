import json
from pathlib import Path
import random

SAMPLE_PAIRS = [
    # 1. Nấu ăn & Tiết kiệm
    ("Tối nay6 tôi tự nấu ăn ở nhà để tiet kiệm fiền", "Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền."),
    ("Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền.", "Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền."),
    ("Tối nay tôi tự nấu an ở nhà để tiết kiệm tiền.", "Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền."),
    ("tối nay tôi tự nấu ăn ở nhà để tiet kiệm tiền", "Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền."),
    ("Tối nay tôi tự nấu ăn ở nhà đễ tiếts kiệm tiền.", "Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền."),
    ("Tối nay tôi tự nâu ăn ở nhà để itết kiệm tiền.", "Tối nay tôi tự nấu ăn ở nhà để tiết kiệm tiền."),

    # 2. Sở thích tiếng Anh
    ("sở thích của tôu là học tiếng anh", "Sở thích của tôi là học tiếng Anh."),
    ("sở thích của tâu là học tiếng anh", "Sở thích của tôi là học tiếng Anh."),
    ("sở thích của tou là học tiếng anh", "Sở thích của tôi là học tiếng Anh."),
    ("Sở thích của tôi là học tiếng Anh.", "Sở thích của tôi là học tiếng Anh."),
    ("so thich cua toi la hoc tieng anh", "Sở thích của tôi là học tiếng Anh."),
    ("sở thích của tôi là học tiếng anh và nghiên cứu ai", "Sở thích của tôi là học tiếng Anh và nghiên cứu AI."),

    # 3. Nghỉ học vì mưa
    ("Hôm nayw tôi không đi họt đi trời jưa", "Hôm nay tôi không đi học vì trời mưa."),
    ("Hôm nayw tôi không đi họt dĩ trời jưa", "Hôm nay tôi không đi học vì trời mưa."),
    ("Hôm nay tôi không đi học vì trời mưa.", "Hôm nay tôi không đi học vì trời mưa."),
    ("hôm nay tôi không đi họt vì trời mưa", "Hôm nay tôi không đi học vì trời mưa."),
    ("Hôm nay tôi không đi học vì trời mưa to.", "Hôm nay tôi không đi học vì trời mưa to."),
    ("Tôi không đi học vì trời mưa to.", "Tôi không đi học vì trời mưa to."),

    # 4. ĐH Bách Khoa
    ("Trường dạy học Bách Khoa co về dày lịch sử và truyền thongw vẽ vang.", "Trường đại học Bách Khoa có bề dày lịch sử và truyền thống vẻ vang."),
    ("Trường đại học Bách Khoa có bề dày lịch sử và truyền thống vẻ vang.", "Trường đại học Bách Khoa có bề dày lịch sử và truyền thống vẻ vang."),
    ("trường đại học bách khoa hà nội đào tạo kỹ sư giỏi", "Trường đại học Bách Khoa Hà Nội đào tạo kỹ sư giỏi."),
    ("Trường đại học Bách Khoa Hà Nội đào tạo kỹ sư công nghệ thông tin giỏi.", "Trường đại học Bách Khoa Hà Nội đào tạo kỹ sư công nghệ thông tin giỏi."),

    # 5. Học sâu & Nơ-ron
    ("Mô hình uọc sâu cà ma5ng nơ-ronj biên đổi cách fon người giải quyết bàu toán phức tạo.", "Mô hình học sâu và mạng nơ-ron biến đổi cách con người giải quyết bài toán phức tạp."),
    ("Mô hình học sâu và mạng nơ-ron biến đổi cách con người giải quyết bài toán phức tạp.", "Mô hình học sâu và mạng nơ-ron biến đổi cách con người giải quyết bài toán phức tạp."),
    ("Mô hình học sâu và mạng nơ-ron mang lại hiệu quả cao.", "Mô hình học sâu và mạng nơ-ron mang lại hiệu quả cao."),
    ("Học phần học sâu giúp sinh viên nắm vững kiến thức mạng nơ-ron.", "Học phần học sâu giúp sinh viên nắm vững kiến thức mạng nơ-ron."),

    # 6. Teencode & Thích đọc
    ("Tôi dất thik đocj truyện tranh vào buỏi chiều tối", "Tôi rất thích đọc truyện tranh vào buổi chiều tối."),
    ("Tôi rất thích đọc truyện tranh và nghe nhạc vào buổi chiều tối.", "Tôi rất thích đọc truyện tranh và nghe nhạc vào buổi chiều tối."),
    ("Tôi rất thích đọc sách nghiên cứu ngôn ngữ.", "Tôi rất thích đọc sách nghiên cứu ngôn ngữ."),

    # 7. Thời tiết miền Bắc
    ("Thời tiếc miền bắc mùa đông xe lạnh và có mưa fùn nhẹ.", "Thời tiết miền Bắc mùa đông se lạnh và có mưa phùn nhẹ."),
    ("Thời tiết miền Bắc mùa đông se lạnn va xó mu7a phùn nhẹ.", "Thời tiết miền Bắc mùa đông se lạnh và có mưa phùn nhẹ."),
    ("Thời tiết miền Bắc mùa đông se lạnh và có mưa phùn nhẹ.", "Thời tiết miền Bắc mùa đông se lạnh và có mưa phùn nhẹ."),
    ("Thời tiết miền Bắc đang bước vào những ngày đầu đông se lạnh.", "Thời tiết miền Bắc đang bước vào những ngày đầu đông se lạnh."),

    # 8. Đọc sách thư viện
    ("Hôm nay trời rất trong xabh, tôi ngồi đocj sasch tại ghư viện.", "Hôm nay trời rất trong xanh, tôi ngồi đọc sách tại thư viện."),
    ("Hôm nay trời rất trong xanh, tôi ngồi đọc sách tại thư viện thành phố.", "Hôm nay trời rất trong xanh, tôi ngồi đọc sách tại thư viện thành phố."),
    ("Hôm nay trời rất trong xanh mát mẻ.", "Hôm nay trời rất trong xanh mát mẻ."),
]

def main():
    random.seed(42)
    augmented = []
    
    # Nhân bản để tập huấn luyện vừa đủ hội tụ nhanh trên CPU (khoảng 100-120 mẫu)
    for rep in range(3):
        for inp, tgt in SAMPLE_PAIRS:
            augmented.append({
                "id": f"sample_{len(augmented):05d}",
                "input_text": inp,
                "target_text": tgt
            })
    
    random.shuffle(augmented)
    
    n = len(augmented)
    train_n = int(n * 0.85)
    train_data = augmented[:train_n]
    val_data = augmented[train_n:]
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open("data/processed/val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open("data/processed/test.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[*] Đã tạo thành công {len(train_data)} mẫu train và {len(val_data)} mẫu val chuẩn xác!")

if __name__ == "__main__":
    main()
