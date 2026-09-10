# 基于RAG的本地论文文献智能问答系统
课程设计项目，实现本地PDF论文知识库问答，支持回答溯源PDF页码。

## 📝项目简介
针对阅读PDF论文信息分散、查找原文困难问题，构建本地化RAG问答工具。
- PDF论文加载解析
- 滑动窗口文本分块 chunk_size=500 overlap=50
- Chroma向量数据库语义检索，相似度阈值过滤
- RAG检索问答，回答附带原文页码溯源
- Streamlit网页前端

## 🛠技术栈
Python、LangChain、ChromaDB、Streamlit、OpenAI API、PyPDF

## 📦安装依赖
```bash
pip install -r requirements.txt
<img width="1910" height="915" alt="网页运行图片" src="https://github.com/user-attachments/assets/cb29daef-716b-445e-a660-dee18185bacb" />
