# Tóm Tắt Các Độ Đo Để Trình Bày

Tài liệu này tóm tắt 3 độ đo vừa được đưa vào pipeline evaluation:

1. `Diversity Openings`
2. `Human Raw Agreement`
3. `Human Gwet's AC1`

Mục tiêu của tài liệu là giải thích:

- độ đo này đo cái gì,
- công thức tính toán là gì,
- vì sao nó hợp lý cho bài toán sinh câu hỏi trắc nghiệm,
- và cách diễn giải kết quả khi trình bày với giáo sư.

---

## 1. Diversity Openings

### Mục tiêu

Đo mức độ đa dạng của cách mở đầu câu hỏi do AI sinh ra.

Ý tưởng chính:

- Nếu nhiều câu hỏi cùng bắt đầu bằng một vài template lặp lại, bộ câu hỏi sẽ nghe "cứng" và ít tự nhiên.
- Nếu các cách mở đầu được phân bố đều hơn, bộ câu hỏi sẽ đa dạng hơn về mặt diễn đạt.

Metric này không đo "chất lượng nội dung" của câu hỏi. Nó chỉ đo "độ đa dạng hình thức mở đầu".

### Cách xây dựng metric

Với mỗi câu hỏi, ta lấy `opening signature`:

- chuẩn hóa văn bản,
- lấy $m$ token đầu tiên,
- trong implementation hiện tại: $m = 3$.

Ví dụ:

- "Đặc điểm nào sau đây..." -> opening signature = `đặc điểm nào`
- "Phát biểu nào đúng..." -> opening signature = `phát biểu nào`

Sau đó, trên toàn bộ tập `N` câu hỏi:

- gom các opening signatures khác nhau,
- đếm tần suất của mỗi signature,
- chuyển sang phân phối xác suất.

Ký hiệu:

- $c_j$: số câu hỏi có opening signature thứ $j$
- $N$: tổng số câu hỏi
- $p_j = \dfrac{c_j}{N}$
- $K$: số opening signatures phân biệt

### Công thức toán học

Shannon entropy của phân phối openings:

$$
H = - \sum_{j=1}^{K} p_j \log_2 p_j
$$

Entropy chuẩn hóa:

$$
H_{\mathrm{norm}} = \frac{H}{\log_2 K}
$$

Score chính được báo cáo:

$$
\mathrm{Opening\ Diversity\ Score} = 100 \times H_{\mathrm{norm}}
$$

Ngoài ra còn có 2 chỉ số phụ:

Số openings hiệu dụng:

$$
\mathrm{Effective\ Number\ of\ Openings} = 2^H
$$

Tỉ lệ lặp lại:

$$
\mathrm{Repetition\ Rate} = 100 \times \frac{\sum_{j=1}^{K} \max(c_j - 1, 0)}{N}
$$

### Diễn giải bằng lời

- Nếu tất cả câu hỏi bắt đầu giống nhau, entropy gần $0$, score gần $0\%$.
- Nếu nhiều cách mở đầu cùng xuất hiện và phân bố khá đều, entropy cao, score gần $100\%$.
- `Effective Number of Openings` trả lời câu hỏi: "Xét theo tần suất thực tế, bộ câu hỏi này tương đương với bao nhiêu kiểu mở đầu có ý nghĩa?"
- `Repetition Rate` cho biết mức độ bị lặp template.

### Vì sao metric này hợp lý

Metric này được lấy cảm hứng từ các hướng đo diversity trong NLG/QG:

- thay vì hard-code danh sách "cụm mở đầu yếu",
- ta đo trực tiếp mức độ tập trung hay phân tán của opening patterns,
- nhưng vẫn giữ được tính dễ giải thích và dễ báo cáo.

Nó phù hợp hơn cho bài toán này vì:

- mục tiêu là đánh giá bộ output đã sinh xong,
- không phải là huấn luyện model bằng loss,
- và không cần một "target distribution" cố định cho openings.

### Câu nói ngắn gọn để trình bày

> Diversity Openings đo mức độ đa dạng của cách bắt đầu câu hỏi bằng entropy chuẩn hóa trên phân phối các opening patterns. Điểm cao có nghĩa là AI không lặp lại một vài khuôn mở đầu cố định, mà phân bố cách diễn đạt đa dạng hơn.

---

## 2. Human Raw Agreement

### Mục tiêu

Đo mức độ đồng thuận trực tiếp giữa các reviewer người khi đánh giá `overall_valid`.

Trong bài toán hiện tại:

- mỗi câu hỏi có 4 reviewer,
- mỗi reviewer cho 1 nhãn nhị phân: `accept` hoặc `reject`.

### Cách tính

Với mỗi câu hỏi `i`, ký hiệu:

- $n$: số reviewer, ở đây $n = 4$
- $n_{i,\mathrm{accept}}$: số reviewer chọn `accept`
- $n_{i,\mathrm{reject}}$: số reviewer chọn `reject`

Độ đồng thuận của câu hỏi `i`:

$$
P_i =
\frac{
\left(n_{i,\mathrm{accept}}^2 + n_{i,\mathrm{reject}}^2\right) - n
}{
n(n - 1)
}
$$

Raw agreement toàn bộ tập:

$$
P_o = \frac{1}{N} \sum_{i=1}^{N} P_i
$$

Trong code, metric được báo cáo là:

$$
\mathrm{Human\ Raw\ Agreement} = 100 \times P_o
$$

### Diễn giải bằng lời

- Nếu 4 reviewer đồng ý hoàn toàn trên mọi câu hỏi, $\mathrm{Raw\ Agreement} = 100\%$.
- Nếu reviewer hay bất đồng, giá trị này giảm xuống.

Raw agreement dễ hiểu nhất vì nó trả lời trực tiếp:

> "Tỉ lệ đồng ý quan sát được giữa các reviewer là bao nhiêu?"

### Điểm mạnh và giới hạn

Điểm mạnh:

- rất dễ giải thích,
- rất hợp để trình bày với audience non-technical,
- phản ánh trực tiếp mức độ đồng thuận thực tế.

Giới hạn:

- không hiệu chỉnh cho đồng thuận do ngẫu nhiên,
- vì vậy nếu một nhãn chiếm ưu thế rất mạnh, raw agreement có thể cao.

### Câu nói ngắn gọn để trình bày

> Human Raw Agreement là tỉ lệ đồng thuận quan sát trực tiếp giữa các reviewer trên nhãn overall_valid. Đây là chỉ số đơn giản và dễ hiểu nhất để mô tả reviewers có đang đánh giá giống nhau hay không.

---



## Tham khảo ngắn gọn

- Sultan et al. (2020), question generation diversity:  
  https://aclanthology.org/2020.acl-main.500/
- Li et al. (2016), diversity-promoting objective / Distinct-style intuition:  
  https://aclanthology.org/N16-1014/
- Liu et al. (2022), phân tích lại diversity metrics:  
  https://aclanthology.org/2022.acl-short.86/
- Wongpakaran et al. (2013), giải thích và ứng dụng Gwet's AC1:  
  https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-61
- Discussion on AC1 vs kappa:  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10205778/
