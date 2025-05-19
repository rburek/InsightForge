# insightforge_helpers.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load environment variables
to_load = os.path.join(os.path.dirname(__file__), '.env')
from dotenv import load_dotenv
load_dotenv(to_load)

# --- LangChain Imports ---
from langchain.evaluation.qa.eval_chain import QAEvalChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# --- Data Loading ---
# Load sales data into a DataFrame
df = pd.read_csv("Datasets/sales_data.csv")

# --- Data Analysis Helpers ---

def advanced_summary():
    return df.describe()


def plot_product_category_sales():
    fig, ax = plt.subplots()
    df.groupby('Product')['Sales'].sum().plot(kind='bar', ax=ax)
    ax.set_title('Sales by Product Category')
    ax.set_xlabel('Product')
    ax.set_ylabel('Total Sales')
    return fig


def plot_sales_trend():
    fig, ax = plt.subplots()
    df.groupby('Date')['Sales'].sum().plot(kind='line', ax=ax)
    ax.set_title('Daily Sales Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    return fig


def plot_product_performance_over_time():
    fig, ax = plt.subplots()
    for product in df['Product'].unique():
        series = df[df['Product'] == product].groupby('Date')['Sales'].sum()
        series.plot(ax=ax, label=product)
    ax.set_title('Product Performance Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    return fig


def plot_regional_sales():
    fig, ax = plt.subplots()
    if 'Region' in df.columns:
        df.groupby('Region')['Sales'].sum().plot(kind='bar', ax=ax)
        ax.set_title('Sales by Region')
        ax.set_xlabel('Region')
        ax.set_ylabel('Total Sales')
    else:
        ax.text(0.5, 0.5, "Column 'Region' not found.", ha='center', va='center')
        ax.set_axis_off()
    return fig


def plot_customer_segmentation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if 'Customer_Gender' in df.columns:
        df['Customer_Gender'].value_counts().plot(kind='bar', ax=axes[0])
        axes[0].set_title('Segmentation by Gender')
        axes[0].set_xlabel('Gender')
        axes[0].set_ylabel('Count')
    else:
        axes[0].text(0.5, 0.5, "Column 'Customer_Gender' not found.", ha='center', va='center')
        axes[0].set_axis_off()
    if 'Customer_Age' in df.columns:
        bins = [0, 18, 30, 45, 60, 100]
        labels = ['<18', '18-29', '30-44', '45-59', '60+']
        df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)
        df['Age_Group'].value_counts().sort_index().plot(kind='bar', ax=axes[1])
        axes[1].set_title('Segmentation by Age Group')
        axes[1].set_xlabel('Age Group')
        axes[1].set_ylabel('Count')
    else:
        axes[1].text(0.5, 0.5, "Column 'Customer_Age' not found.", ha='center', va='center')
        axes[1].set_axis_off()
    plt.tight_layout()
    return fig

# --- AI Chat Chain Setup ---
chat_model = ChatOpenAI(temperature=0)
agent_chain = create_pandas_dataframe_agent(
    llm=chat_model,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)

# --- RAG Insight Setup ---
loader = PyPDFLoader
pdf_dir = "Datasets/PDF Folder"
documents = []
for fname in os.listdir(pdf_dir):
    if fname.lower().endswith('.pdf'):
        path = os.path.join(pdf_dir, fname)
        documents.extend(loader(path).load())
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conv_rag = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

def generate_rag_insight(question: str) -> str:
    result = conv_rag({"question": question})
    return result.get("answer", "")

# --- Model Evaluation ---
def evaluate_model(qa_pairs, threshold: float = 0.8):
    """
    Evaluate a list of QA pairs using QAEvalChain and embedding similarity threshold.
    Each qa_pair should be {'question': str, 'answer': str}.
    Returns a list of dicts with question, predicted, actual, similarity_score, and correctness.
    """
    # 1) Initialize the evaluation chain
    eval_chain = QAEvalChain.from_llm(chat_model, handle_parsing_errors=True)

    # 2) Generate raw predictions
    raw_preds = [
        agent_chain.run(input=q["question"], chat_history="", agent_scratchpad="")
        for q in qa_pairs
    ]

    # 3) Prepare predictions for QAEvalChain (it expects a dict with 'prediction')
    eval_preds = [{"prediction": pred} for pred in raw_preds]

    # 4) Run the chain-based evaluation (optional, can be used for deeper analysis)
    raw_results = eval_chain.evaluate(
        examples=qa_pairs,
        predictions=eval_preds,
        question_key="question",
        answer_key="answer",
        prediction_key="prediction"
    )

    # 5) Compute cosine similarity between embeddings of actual vs predicted
    emb = OpenAIEmbeddings()
    actual_texts = [q["answer"] for q in qa_pairs]
    actual_embs = emb.embed_documents(actual_texts)
    pred_embs = emb.embed_documents(raw_preds)
    sim_scores = []
    for a_vec, p_vec in zip(actual_embs, pred_embs):
        score = np.dot(a_vec, p_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(p_vec))
        sim_scores.append(score)

    # 6) Compile results based on similarity threshold
    eval_results = []
    for q, pred, actual, raw_res, score in zip(qa_pairs, raw_preds, actual_texts, raw_results, sim_scores):
        is_correct = score >= threshold
        eval_results.append({
            "question":         q["question"],
            "predicted":        pred,
            "actual":           actual,
            "similarity_score": score,
            "correct":          is_correct
        })
    return eval_results

# --- Execution Time Monitoring ---
class ModelMonitor:
    def __init__(self): self.logs = []
    def log_interaction(self, question: str, execution_time: float):
        self.logs.append({"timestamp": datetime.now().isoformat(), "question": question, "execution_time": execution_time})
    def get_average_execution_time(self) -> float:
        times = [log['execution_time'] for log in self.logs]
        return sum(times) / len(times) if times else 0.0

model_monitor = ModelMonitor()