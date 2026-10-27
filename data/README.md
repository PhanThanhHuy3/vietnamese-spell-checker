# Quản lý Dữ liệu (Data Management)

Thư mục này dùng để lưu trữ và quản lý dữ liệu huấn luyện cho bài toán Sửa lỗi chính tả Tiếng Việt.

## 1. Cấu trúc thư mục
- `raw/`: Chứa dữ liệu văn bản thô (clean text từ báo chí, Wikipedia tiếng Việt, v.v.) chưa qua xử lý hoặc các bộ dữ liệu gốc thu thập được.
- `synthetic/`: Chứa dữ liệu được sinh nhiễu nhân tạo (tạo lỗi telex, lỗi phương ngữ, gõ sai bàn phím, thiếu dấu) từ dữ liệu sạch.
- `processed/`: Chứa dữ liệu đã qua tiền xử lý, định dạng chuẩn (train.jsonl, val.jsonl, test.jsonl) sẵn sàng đưa vào mô hình.

## 2. Định dạng dữ liệu chuẩn (JSON Lines - `.jsonl`)
Mỗi dòng trong file dữ liệu là một JSON object:
```json
{
  "id": "sample_001",
  "input_text": "Hôm nay chời mưa to quá, tôi quến mang ô.",
  "target_text": "Hôm nay trời mưa to quá, tôi quên mang ô."
}
```

## 3. Các loại lỗi chính tả tiếng Việt thường gặp:
1. **Lỗi gõ Telex / Unikey**: `hoangf` -> `hoàng` (lỗi không nhận dấu), `nhaf` -> `nhà`, `thìe` -> `thì`.
2. **Lỗi phương ngữ (Homophones / Regional variants)**:
   - Phụ âm đầu: `ch` <-> `tr`, `s` <-> `x`, `d` <-> `gi` <-> `r`, `l` <-> `n`.
   - Phụ âm cuối: `c` <-> `t` (`manh mẽ` -> `mạnh mẽ`, `bắt` -> `bắc`), `n` <-> `ng` (`buồn` -> `buồng`).
   - Dấu thanh: `hỏi` <-> `ngã` (`sữa` <-> `sửa`, `nghĩ` <-> `nghỉ`).
3. **Lỗi gõ phím lân cận (Keyboard Distance - QWERTY)**:
   - Gõ nhầm phím kế bên (ví dụ: `a` nhầm sang `s`, `q`, `w`).
4. **Lỗi dấu thanh và nguyên âm**:
   - Thiếu dấu hoặc đặt sai vị trí dấu (`hoà` vs `hòa`, `thuyển` vs `thuyền`).
