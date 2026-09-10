import os
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ===================== Pydantic 数据模型校验 =====================
class PaperQueryInput(BaseModel):
    """用户提问输入校验模型"""
    question: str = Field(min_length=1, max_length=1000, description="用户针对论文的提问")
    similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.6, description="检索相似度阈值")

class SourceReference(BaseModel):
    """溯源引用数据模型"""
    page_number: int
    source_file: str

# ===================== 全局配置 =====================
PAPER_FOLDER = "./papers"
CHROMA_DB_PATH = "./chroma_db"
os.makedirs(PAPER_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Streamlit页面配置
st.set_page_config(page_title="本地论文RAG智能问答系统", layout="wide")
st.title("📄 基于RAG的论文文献智能问答系统")

# API KEY输入框
openai_api_key = st.text_input("请输入OpenAI API‑Key", type="password")
if not openai_api_key:
    st.warning("请填写OpenAI API Key才能继续使用！")
    st.stop()

# 初始化embedding和大模型
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt‑3.5‑turbo", temperature=0, openai_api_key=openai_api_key)

# 向量数据库
vector_store = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding
)

# ===================== PDF上传与知识库构建 =====================
st.subheader("📂 上传PDF论文文件")
uploaded_pdf = st.file_uploader("上传论文PDF", type="pdf")

if uploaded_pdf is not None:
    # 保存pdf到本地papers文件夹
    save_path = os.path.join(PAPER_FOLDER, uploaded_pdf.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"文件 {uploaded_pdf.name} 保存成功！")

    if st.button("🔨 构建向量知识库"):
        try:
            loader = PyPDFLoader(save_path)
            docs = loader.load()
            st.info(f"PDF一共读取 {len(docs)} 页")

            # 滑动窗口分块策略 chunk_size=500 overlap=50
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            split_docs = text_splitter.split_documents(docs)
            st.info(f"文档切分完成，共 {len(split_docs)} 个文本块")

            # 写入向量库
            vector_store.add_documents(split_docs)
            st.success("✅ 向量知识库构建完成，可以开始提问！")

        except Exception as e:
            st.error(f"文档处理异常：{str(e)}")

# ===================== RAG问答链 =====================
st.divider()
st.subheader("💬 论文问答")
threshold = st.slider("相似度检索阈值(过滤低相关片段，越高越严格)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
user_question = st.text_area("请输入针对论文的问题：", placeholder="这篇论文的研究方法是什么？创新点有哪些？")

if st.button("🚀 提问") and user_question.strip():
    # pydantic参数校验
    try:
        input_data = PaperQueryInput(question=user_question.strip(), similarity_threshold=threshold)
    except ValidationError as e:
        st.error(f"输入校验失败：{e}")
        st.stop()

    try:
        # 带分数阈值的检索器
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": input_data.similarity_threshold, "k":4}
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是论文阅读助手，严格根据上下文文档回答用户问题。"
             "如果上下文没有答案，直接回答：文档中没有找到相关信息，不要编造内容。\n"
             "上下文：{context}"),
            ("human", "{input}")
        ])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # 执行RAG
        result = rag_chain.invoke({"input": input_data.question})

        st.markdown("### ✨回答结果")
        st.write(result["answer"])

        # ============溯源输出：打印原文页码===========
        st.markdown("### 📑引用来源溯源")
        sources = result["context"]
        if len(sources) == 0:
            st.info("没有检索到相关原文片段")
        else:
            for idx, doc in enumerate(sources):
                page = doc.metadata.get("page", -1) + 1  # pypdf页码从0开始，+1转为真实页码
                src_file = doc.metadata.get("source", "未知文件")
                ref = SourceReference(page_number=page, source_file=os.path.basename(src_file))
                st.markdown(f"{idx+1}. 文件：`{ref.source_file}` | 页码：**{ref.page_number}**")

    except Exception as err:
        st.error(f"问答运行异常：{str(err)}")

st.divider()
st.caption("基于RAG本地论文文献智能问答系统｜LangChain+Chroma+Streamlit")
