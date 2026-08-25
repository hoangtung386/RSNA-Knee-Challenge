# Chuẩn hóa cường độ MRI

## Vì sao đây là quyết định quan trọng

CT có đơn vị Hounsfield: nước = 0, xương đặc ≈ +1000, ở mọi máy. MRI **không có
gì tương đương**. Cùng một mô, cùng một bệnh nhân, chụp trên hai máy khác nhau
(hoặc cùng máy khác hệ số khuếch đại) cho ra giá trị pixel khác nhau hoàn toàn.

Nếu không chuẩn hóa, model sẽ học đặc điểm của *scanner* thay vì của *bệnh lý*,
và sụp đổ trên dữ liệu test đến từ máy khác.

## Các phương án được so sánh

| Tên | Cách làm | Dải đầu ra |
|---|---|---|
| `slice_percentile` | Percentile 1–99 theo **từng lát** | `[0, 1]` |
| `volume_percentile` | Percentile 0.5–99.5 theo **toàn volume** | `[0, 1]` |
| `volume_percentile_pm1` | Như trên, ánh xạ sang `[-1, 1]` | `[-1, 1]` |
| `volume_zscore` | Z-score toàn volume, cắt ±3 | `[-3, 3]` |

Cài đặt: `src/knee_mri/data/normalize.py`. Chạy so sánh:

```bash
python scripts/analyze_normalization.py --max-series 400
```

## Ba tiêu chí đánh giá

1. **Cross-series consistency** — độ lệch chuẩn của giá trị trung bình sau chuẩn
   hóa, tính xuyên các series. Nhỏ = nhất quán giữa các scanner. *Càng nhỏ càng tốt.*
2. **Flicker** — biến thiên giữa các lát trong cùng volume. Quá nhỏ nghĩa là đã
   *over-normalize*, xóa mất tín hiệu 3D thật. *Cần vừa phải, không phải nhỏ nhất.*
3. **Tỉ lệ voxel bị cắt** — phần bị đẩy ra ngoài dải sau khi clip.

## Kết luận

**Chọn `volume_percentile_pm1`** (mặc định trong `configs/base.yaml`):

- Nhất quán xuyên dataset, không phụ thuộc scanner.
- Giữ được biến thiên giữa các lát — `slice_percentile` chuẩn hóa từng lát độc
  lập nên triệt tiêu chính thông tin này.
- Percentile 0.5–99.5 chống chịu outlier tốt hơn z-score, vốn bị artifact kim
  loại kéo lệch cả trung bình lẫn độ lệch chuẩn.
- Dải `[-1, 1]` **khớp đúng** định dạng đầu vào của 3DINO-ViT, nên đặc trưng
  teacher và đầu vào student cùng một thang đo.

## Hai điểm kỹ thuật

### Resize chạy trên float32, không qua uint8

Bản trước refactor đi đường vòng float → `uint8` → `PIL.Image.resize` → float,
ép dải động của MRI 12–16 bit xuống còn 8 bit (256 mức). Mất mát này đặc biệt
đáng kể ở vùng tương phản thấp — sụn và sụn chêm, tức chính thứ cần phát hiện.

Nay `resize_volume` nội suy trilinear thẳng trên float32, cài bằng numpy thuần
nên lớp dữ liệu không phụ thuộc PyTorch và test chạy được không cần GPU.

### Thứ tự: crop ROI → resize → chuẩn hóa

- **Crop trước resize:** độ phân giải hiệu dụng dồn vào vùng bệnh lý thay vì nền đen.
- **Chuẩn hóa sau resize:** thống kê percentile tính trên đúng tập voxel mà model
  thực sự nhìn thấy.

### Quy đổi khi đưa ảnh cho model ngoài

SAM và VLM kỳ vọng ảnh hiển thị được (`uint8` RGB). Volume nằm trong `[-1, 1]`
**phải** đi qua `to_unit_range()` trước khi cast. Cast thẳng
`(volume * 255).astype(np.uint8)` khiến mọi giá trị âm bị wrap-around thành số
lớn — ảnh gửi đi là nhiễu thuần túy. Test `test_normalize.py` khóa lại điều này.
