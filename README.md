# Ai-fridays
TCS ai friday

https://github.com/Nithindev-sudo/Energy-Consumption/blob/main/app.py



Business analyst agent

import pandas as pd
from typing import List, Dict

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import gradio as gr

# -----------------------------
# Data Processor
# -----------------------------

class DataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_documents(self) -> List[Document]:
        df = pd.read_csv(self.csv_path)

        documents = []
        for _, row in df.iterrows():
            content = f"""
Month: {row['month']}
Department: {row['department']}
Sales (INR): {row['sales_inr']}
Expenses (INR): {row['expenses']}
Customers: {row['customers']}
Inventory Cost (INR): {row['inventory_cost']}
Marketing Spend (INR): {row['marketing_spend_inr']}
"""

            documents.append(
                Document(
                    page_content=content.strip(),
                    metadata={"department": row["department"]}
                )
            )

        return documents


# -----------------------------
# Business Consultant Agent
# -----------------------------

class BusinessAgent:
    def __init__(self, documents: List[Document]):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        self.llm = Ollama(
            model="llama3",
            temperature=0.2
        )

        self.prompt = PromptTemplate(
            template="""
You are a Virtual CFO for a small business in India.

Rules:
- Always calculate numbers in INR.
- Explain insights simply for a non-technical business owner.
- If expenses or inventory cost increased month-over-month, suggest 3 cost-saving actions.
- If marketing spend increased, comment on customer growth efficiency.
- If sales are flat, suggest 2 growth tactics.

Context:
{context}

Question:
{question}

Give actionable, business-friendly advice.
""",
            input_variables=["context", "question"]
        )

    def acl_filter(self, role: str) -> Dict:
        if role == "Sales":
            return {"department": "Sales"}
        elif role == "Finance":
            return {"department": "Finance"}
        elif role == "Owner":
            return {}
        return {"department": "None"}

    def query(self, question: str, role: str) -> str:
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 6,
                "filter": self.acl_filter(role)
            }
        )

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )

        return qa.run(question)


# -----------------------------
# Gradio UI
# -----------------------------

class GradioInterface:
    def __init__(self, agent: BusinessAgent):
        self.agent = agent

    def launch(self):
        def consult(role, question):
            if not question.strip():
                return "Please ask a business-related question."
            return self.agent.query(question, role)

        with gr.Blocks() as app:
            gr.Markdown("## 📈 Virtual Business Consultant (MSME Dashboard)")

            role = gr.Dropdown(
                ["Sales", "Finance", "Owner"],
                label="Login Role"
            )

            question = gr.Textbox(
                lines=3,
                placeholder="Are my expenses growing faster than sales?"
            )

            response = gr.Textbox(lines=12)

            gr.Button("Get Advice").click(
                consult,
                inputs=[role, question],
                outputs=response
            )

        app.launch()


# -----------------------------
# Run App
# -----------------------------

if __name__ == "__main__":
    docs = DataProcessor("business_data.csv").load_documents()
    agent = BusinessAgent(docs)
    GradioInterface(agent).launch()
