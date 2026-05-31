# CÁC CÂU HỎI THI VẤN ĐÁP TIỀM NĂNG & GỢI Ý TRẢ LỜI
*Tập hợp các câu hỏi giảng viên thường đặt ra khi bảo vệ bài tập lớn / thi vấn đáp về phần mạng LSTM 1D và BiLSTM trong nhận diện tin giả.*

---

## CHỦ ĐỀ 1: KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)

### Câu 1: Em hãy giải thích sự khác biệt lớn nhất giữa LSTM 1 chiều (LSTM 1D) và LSTM 2 chiều (BiLSTM)? Trong code em cấu hình hai loại này như thế nào?
* **Gợi ý trả lời**:
  * **Về mặt lý thuyết**: LSTM 1 chiều chỉ quét văn bản theo một hướng thuận từ đầu đến cuối câu (từ trái qua phải). Trạng thái ẩn tại bước $t$ chỉ chứa thông tin ngữ cảnh của các từ đứng trước. BiLSTM quét đồng thời theo 2 hướng độc lập: Xuôi từ trái sang phải ($\overrightarrow{h_t}$) và Ngược từ phải sang trái ($\overleftarrow{h_t}$). Vectơ đặc trưng đầu ra của BiLSTM là sự ghép nối (concatenation) của cả hai chiều: $h_t = [\overrightarrow{h_t} \mathbin{\Vert} \overleftarrow{h_t}]$, giúp tích hợp ngữ cảnh cả trước và sau của mỗi từ.
  * **Về cấu hình trong code**:
    * Trong lớp `BiLSTMClassifier` (tệp [lstm_model.py](file:///d:/Fake-news-detection/src/lstm_model.py)), cấu hình qua tham số `bidirectional` của lớp `nn.LSTM` trong PyTorch:
      * `bidirectional=True` $\rightarrow$ BiLSTM.
      * `bidirectional=False` $\rightarrow$ LSTM 1 chiều.
    * Khi tính toán kích thước đầu vào của tầng tuyến tính phân loại (Fully Connected Layer):
      * Nếu `bidirectional=True`, chiều đầu vào là `hidden_dim * 2`.
      * Nếu `bidirectional=False`, chiều đầu vào là `hidden_dim`.

### Câu 2: Tại sao sau lớp LSTM em lại áp dụng Global Max Pooling 1D (`torch.max(out, dim=1)`) thay vì lấy trạng thái ẩn cuối cùng ($h_n$ hoặc $h_T$)?
* **Gợi ý trả lời**:
  * Đối với các chuỗi văn bản dài, các trạng thái ẩn cuối cùng ($h_T$) của mạng tuần tự dễ bị mất mát thông tin ngữ cảnh của những từ xuất hiện ở đầu câu (hiện tượng tiêu biến gradient / quên thông tin).
  * **Global Max Pooling 1D** quét qua toàn bộ chiều dài câu (`dim=1` tức `seq_len`) để lấy giá trị kích hoạt lớn nhất của các nơ-ron đặc trưng. Mỗi nơ-ron trong LSTM thường nhạy cảm với một nhóm từ khóa nhất định. Trong bài toán nhận diện tin giả, việc áp dụng Max Pooling giúp mô hình bắt giữ được những từ khóa mang tính chất giật gân, phóng đại (như: `tin nóng`, `thần dược`, `sốc`) xuất hiện ở bất kỳ vị trí nào trong câu mà không bị ảnh hưởng bởi độ dài văn bản.

### Câu 3: Lớp Embedding của em có tham số `padding_idx=0` để làm gì?
* **Gợi ý trả lời**:
  * `padding_idx=0` chỉ định cho PyTorch biết rằng chỉ số 0 tương ứng với token đệm `<pad>`.
  * Vectơ nhúng của token `<pad>` tại chỉ số này sẽ luôn được cố định bằng 0 và quan trọng là **không cập nhật gradient** trong suốt quá trình lan truyền ngược. Điều này giúp mô hình bỏ qua, không học ngữ nghĩa từ các ký tự trắng/token đệm dùng để làm đều độ dài câu.

---

## CHỦ ĐỀ 2: TIỀN XỬ LÝ & TOKENIZATION (DATA PREPROCESSING)

### Câu 4: Tại sao em phải thực hiện tách từ ghép Tiếng Việt (Word Segmentation) bằng thư viện `pyvi`? Nếu không thực hiện thì mô hình LSTM có ảnh hưởng gì không?
* **Gợi ý trả lời**:
  * Tiếng Việt là ngôn ngữ đơn âm tiết nhưng đa phần ý nghĩa nằm ở các từ phức/từ ghép (như `thành phố`, `học sinh`, `tin giả`). 
  * Nếu tách từ theo khoảng trắng thông thường (whitespace tokenization), mô hình sẽ hiểu `tin` và `giả` là hai từ độc lập, làm mất đi ngữ nghĩa của từ ghép `tin_giả`.
  * Sử dụng `pyvi` để liên kết từ ghép bằng dấu gạch dưới (ví dụ: `tin giả` $\rightarrow$ `tin_giả`) giúp bộ từ điển `Vocab` coi từ ghép này là một token duy nhất. Từ đó, lớp Embedding của LSTM sẽ học được vectơ biểu diễn chính xác cho ngữ nghĩa của từ ghép đó, giúp mô hình hoạt động hiệu quả hơn.

### Câu 5: Bộ từ điển `Vocab` tự xây dựng của em giải quyết vấn đề từ lạ (Out-Of-Vocabulary - OOV) ở tập Test hoặc lúc chạy thực tế như thế nào?
* **Gợi ý trả lời**:
  * Khi xây dựng `Vocab` trên tập huấn luyện, em khởi tạo một token đặc biệt là `<unk>` (Unknown) có chỉ số là 1.
  * Trong hàm `encode` của [data_loader.py](file:///d:/Fake-news-detection/src/data_loader.py):
    ```python
    idxs = [self.word2idx.get(tok, self.word2idx["<unk>"]) for tok in tokens]
    ```
  * Bất kỳ từ nào xuất hiện ở tập Test hoặc do người dùng nhập vào mà không nằm trong từ điển huấn luyện sẽ được tự động ánh xạ về chỉ số của `<unk>` (index 1). Điều này giúp mô hình không bị lỗi và vẫn thực hiện dự đoán bình thường.

---

## CHỦ ĐỀ 3: QUY TRÌNH HUẤN LUYỆN & REGULARIZATION (TRAINING)

### Câu 6: Làm thế nào em giải quyết vấn đề mất cân bằng lớp (tin thật nhiều hơn tin giả) trong tập dữ liệu của mình?
* **Gợi ý trả lời**:
  * Em triển khai hai giải pháp độc lập và loại trừ lẫn nhau:
    1. **Random Oversampling (Lặp mẫu ngẫu nhiên)**: Sao chép ngẫu nhiên các mẫu dữ liệu của lớp tin giả (lớp thiểu số) trong tập huấn luyện cho đến khi số lượng mẫu của hai lớp bằng nhau.
    2. **Class Weights (Trọng số lớp)**: Tính toán trọng số tỉ lệ nghịch với tần suất xuất hiện của lớp: $W_c = N_{\text{total}} / (2 \times N_c)$ rồi truyền trọng số này vào hàm Loss: `nn.CrossEntropyLoss(weight=class_weights)`. Khi đó, nếu mô hình đoán sai tin giả, nó sẽ bị phạt nặng hơn so với việc đoán sai tin thật.
  * Hai kỹ thuật này phải loại trừ lẫn nhau (oversample=True thì tắt class_weights và ngược lại) để tránh việc mô hình thiên vị quá mức cho lớp thiểu số.

### Câu 7: Em áp dụng những kỹ thuật nào để chống quá khớp (Overfitting) cho mô hình LSTM?
* **Gợi ý trả lời**:
  * LSTM rất dễ bị overfitting trên dữ liệu văn bản nhỏ, vì vậy em đã áp dụng:
    1. **L2 Regularization (Weight Decay = 1e-4)**: Tích hợp vào bộ tối ưu Adam để phạt các trọng số có giá trị quá lớn, giảm độ phức tạp của mô hình.
    2. **Dropout (Tỷ lệ 0.3)**: Áp dụng sau lớp Embedding (Embedding Dropout) để tránh việc mô hình ghi nhớ máy móc các từ khóa cụ thể, và áp dụng trong Classification Head.
    3. **ReduceLROnPlateau (LR Scheduler)**: Tự động giảm tốc độ học đi một nửa nếu Validation F1-score không cải thiện sau 2 epoch để giúp mô hình hội tụ tốt hơn.
    4. **Early Stopping**: Giám sát Validation F1-score; nếu liên tiếp trong 5 epoch (`patience=5`) không có sự cải thiện, mô hình sẽ tự động dừng huấn luyện để tránh việc học tủ tập Train.

---

## CHỦ ĐỀ 4: THUẬT TOÁN GIẢI THÍCH (EXPLAINABILITY)

### Câu 8: Em hãy trình bày nguyên lý hoạt động của thuật toán giải thích từ khóa (Occlusion) được dùng cho LSTM? Tại sao em che từ bằng `<pad>` mà không xóa hẳn từ đó đi?
* **Gợi ý trả lời**:
  * **Nguyên lý**: Thuật toán lần lượt che (mask) từng từ ứng viên trong câu bằng token `<pad>` (index 0), sau đó đưa câu bị che vào mô hình để dự đoán lại xác suất lớp đúng. Điểm đóng góp của từ đó được tính bằng sự sụt giảm xác suất: $\text{Score}(i) = P_{\text{gốc}} - P_{\text{sau\_khi\_che\_từ\_i}}$. Từ nào bị che mà làm xác suất nhãn đúng giảm mạnh nhất ($\text{Score}$ lớn nhất) thì từ đó đóng vai trò quan trọng nhất giúp mô hình ra quyết định.
  * **Tại sao che bằng `<pad>` thay vì xóa từ**:
    * Mạng tuần tự LSTM xử lý chuỗi dựa trên bước thời gian và vị trí tuyệt đối của các từ.
    * Nếu ta xóa từ đó ra khỏi câu, độ dài câu bị co lại, làm dịch chuyển vị trí của toàn bộ các từ phía sau lên trước, gây xáo trộn ngữ cảnh truyền qua trạng thái ẩn của LSTM.
    * Việc che bằng cách ghi đè token `<pad>` giúp giữ nguyên độ dài chuỗi và vị trí tuyệt đối của các từ khác, đảm bảo tính nhất quán và loại bỏ nhiễu lệch vị trí.
