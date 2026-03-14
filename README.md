# 📄 Advanced Document Retrieval System

An intelligent, high-performance RAG (Retrieval-Augmented Generation) system for PDF documents. Built with a modern **React** frontend and a robust **FastAPI** backend, featuring open-source document extraction, hybrid search, and cross-encoder reranking.

---

## 🚀 Key Features

*   **Modern React UI**: A responsive, premium dashboard for document management and intelligent chat.
*   **Open-Source Extraction**: Leverages **Docling** for high-fidelity, structure-aware PDF parsing.
*   **Hybrid Search Engine**: Combines **FAISS** (Vector Search) and **BM25** (Lexical Search) with **Reciprocal Rank Fusion (RRF)** for superior retrieval precision.
*   **Intelligent Reranking**: Optional **Cross-Encoder Reranker** (HuggingFace) to fine-tune results and maximize answer accuracy.
*   **Semantic Routing**: Automatic query routing to specific document sections based on content type.
*   **Detailed Analytics**: Real-time stats on processing time, chunk counts, and retrieval confidence.

---

## 🏗️ Architecture

```mermaid
graph TD
    User[👤 User] -->|Interacts| React["⚛️ React Frontend<br>(frontend/)"]
    
    subgraph API_Layer [Backend API]
        React -->|HTTP / JSON| FastAPI["⚡ FastAPI Backend<br>(backend/main.py)"]
        FastAPI -->|Query/Upload| Store["📦 Document Store<br>(EnhancedDocumentStoreHybrid)"]
    end
    
    subgraph Processing_Layer [Ingestion & Processing]
        Store -->|Extract| Docling["📄 Docling<br>(Open-Source PDF Extraction)"]
        Store -->|Chunk| Chunker["✂️ Chunker<br>(Logical Boundaries)"]
    end

    subgraph Retrieval_Layer [Hybrid Search & RAG]
        Store -->|Retrieve| Hybrid["🔍 Hybrid Retriever<br>(FAISS + BM25)"]
        Hybrid -->|Fusion| RRF["⚖️ RRF Scoring"]
        RRF -->|Rank| Reranker["⭐ Cross-Encoder Rerank"]
        Reranker -->|Context| LLM["🤖 Gemini LLM<br>(Answer Generation)"]
    end

    subgraph Data_Storage [Local Storage]
        Hybrid -->|FAISS Index| VectorDB[(Vector Store)]
        Hybrid -->|BM25 Index| DocDB[(Lexical Store)]
    end
```

---

## 🛠️ Setup & Execution

### 1. Requirements
* **Node.js**: For the React frontend.
* **Python 3.10+**: For the FastAPI backend.
* **Gemini API Key**: For answer generation.

### 2. Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure Environment: Create a `.env` file in the root or backend folder:
    ```text
    GOOGLE_API_KEY=your_gemini_api_key
    ```
4.  Run Backend:
    ```bash
    uvicorn main:app --port 8000
    ```

### 3. Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run Dev Server:
    ```bash
    npm run dev
    ```
4.  Open `http://localhost:5173` in your browser.

---

## 📈 Performance & Evaluation

The system's performance is validated using the **Ragas** evaluation framework, focusing on faithfulness, relevancy, and retrieval quality.

### Evaluation Metrics (Latest Run)
*   **Faithfulness**: **0.84** (High adherence to the source document)
*   **Answer Relevancy**: **0.86** (Measures how pertinent the answer is to the query)
*   **Context Precision**: **0.88** (Quality of the retrieved chunks)
*   **Context Recall**: **0.976** (Ability to retrieve all relevant information)

> [!NOTE]
> Evaluation was performed on a diverse set of complex financial and legal documents to ensure robustness across different domains.

---

## 📂 Project Structure

*   **`frontend/`**: Vite + React application.
    *   `src/App.jsx`: Main UI logic and chat interface.
*   **`backend/`**: FastAPI server.
    *   `main.py`: API endpoints and store orchestration.
    *   `core/`: Core RAG logic (Retriever, Chunker, PDF Processor).
*   **`notebooks/`**: Evaluation and experimentation notebooks.