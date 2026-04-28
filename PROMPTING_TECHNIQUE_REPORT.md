# Báo Cáo Ngắn Về Kỹ Thuật Prompting Trong `explain_mcq.py`

## 1. Kỹ Thuật Đang Sử Dụng

Trong module `src/gen/explain_mcq.py`, kỹ thuật prompting chính có thể mô tả là:

**Retrieval-Augmented, Citation-Grounded Instruction Prompting**

Nói ngắn gọn, mô hình không sinh giải thích hoàn toàn dựa trên tri thức nội tại. Thay vào đó, hệ thống truy hồi các đoạn nội dung liên quan từ slide và transcript bài giảng, đưa chúng vào prompt làm ngữ cảnh, rồi yêu cầu LLM sinh lời giải thích có cấu trúc và có trích dẫn nguồn.

## 2. Cách Triển Khai Trong Code

Prompt trong `explain_mcq.py` kết hợp các thành phần sau:

- **Role prompting**: gán vai trò cho mô hình là giảng viên đại học chuyên giải thích câu hỏi trắc nghiệm.
- **Instruction prompting**: yêu cầu mô hình giải thích vì sao đáp án đúng là đúng và vì sao các distractor là sai.
- **Task decomposition**: chia lời giải thích thành 4 phần: lý do ra câu hỏi, giải thích đáp án đúng, giải thích distractor, và ngữ cảnh kiến thức.
- **Retrieval-augmented prompting**: trước khi gọi LLM, hệ thống truy hồi các context blocks liên quan từ slide PDF và video transcript bằng hybrid retrieval, sau đó đưa vào prompt.
- **Citation-grounded prompting**: prompt bắt buộc gắn trích dẫn slide, số trang, và YouTube URL nếu có.
- **Structured output prompting**: prompt yêu cầu mô hình chỉ trả về JSON theo schema cố định.

Vì vậy, đây là một dạng prompt được "ground" vào học liệu chính thức của môn học, thay vì chỉ là prompt hỏi đáp thông thường.

## 3. Cơ Sở Khoa Học

Có thể giải thích kỹ thuật này dựa trên ba hướng nghiên cứu chính:

1. **Retrieval-Augmented Generation (RAG)**

   Bài báo: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* - Lewis et al., NeurIPS 2020.  
   Link: https://arxiv.org/abs/2005.11401

   Bài báo này đề xuất việc kết hợp mô hình sinh ngôn ngữ với nguồn tri thức bên ngoài được truy hồi từ kho tài liệu. Trong dự án, `explain_mcq.py` áp dụng ý tưởng tương tự: truy hồi slide/transcript liên quan, rồi dùng chúng làm ngữ cảnh để sinh giải thích.

2. **Citation-Aware / Citation-Grounded Generation**

   Bài báo: *Enabling Large Language Models to Generate Text with Citations* - Gao et al., EMNLP 2023.  
   Link: https://arxiv.org/abs/2305.14627

   Bài báo này tập trung vào việc giúp LLM sinh văn bản có trích dẫn để tăng tính kiểm chứng và giảm hallucination. Trong dự án, mỗi lời giải thích được yêu cầu gắn với slide, số trang hoặc video bài giảng.

3. **Prompt Pattern / Instruction Pattern**

   Bài báo: *A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT* - White et al., 2023.  
   Link: https://arxiv.org/abs/2302.11382

   Bài báo này mô tả các mẫu prompt như role/persona prompting, template prompting, instruction prompting và output-format constraints. `explain_mcq.py` kết hợp các pattern này để điều khiển vai trò, nhiệm vụ và định dạng đầu ra của mô hình.

## 4. Có Phải Chain-of-Thought Prompting Không?

Không nên mô tả module này là **Chain-of-Thought prompting** theo nghĩa chuẩn.

Lý do là prompt có yêu cầu mô hình sinh explanation/rationale, nhưng không yêu cầu mô hình trình bày chuỗi suy luận từng bước theo kiểu "think step by step", và cũng không cung cấp các ví dụ CoT mẫu trong prompt.

Nếu cần đối chiếu, có thể nhắc đến bài báo:

*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* - Wei et al., NeurIPS 2022.  
Link: https://arxiv.org/abs/2201.11903

Tuy nhiên, kỹ thuật đang dùng trong `explain_mcq.py` phù hợp hơn với **RAG + citation-grounded instruction prompting**.

## 5. Kết Luận Đề Xuất Để Trình Bày

Có thể trình bày với giáo sư như sau:

> Trong module `explain_mcq.py`, em sử dụng kỹ thuật **Retrieval-Augmented, Citation-Grounded Instruction Prompting**. Hệ thống truy hồi các đoạn nội dung liên quan từ slide và transcript bài giảng, đưa vào prompt làm ngữ cảnh, sau đó yêu cầu LLM sinh giải thích cho đáp án đúng và các distractor theo cấu trúc JSON, đồng thời bắt buộc kèm trích dẫn nguồn. Cách tiếp cận này dựa trên ý tưởng Retrieval-Augmented Generation của Lewis et al. (2020), kết hợp với hướng sinh văn bản có trích dẫn của Gao et al. (2023), nhằm tăng tính đúng đắn, tính kiểm chứng và giảm hallucination trong giải thích câu hỏi trắc nghiệm.

