import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from llama_cpp import Llama

from config import *

class QueryEngine:
    def __init__(
        self,
        model_path: str,
        vectorstore_path: str,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        top_k: int = 5,
    ):
        self.model_path = model_path
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.top_k = top_k

        # Загружаем vector store
        self.vectorstore = None
        self._load_vectorstore()

        # Загружаем embedding модель для query
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Загружаем локальную модель через llama_cpp
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )

    def _load_vectorstore(self):
        index_file = os.path.join(self.vectorstore_path, "index.faiss")
        pkl_file = os.path.join(self.vectorstore_path, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                HuggingFaceEmbeddings(model_name=self.embedding_model_name),
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(f"Vectorstore not found in {self.vectorstore_path}")

    def query(self, query_text: str) -> str:
        # 1. Найти релевантные документы
        query_embedding = self.embedding_model.embed_query(query_text)
        docs: list[Document] = self.vectorstore.similarity_search_by_vector(query_embedding, k=self.top_k)

        # Собираем контекст из найденных чанков
        context = "\n---\n".join([doc.page_content for doc in docs])

        # 2. Формируем промпт для генерации
        prompt = rf"""
        ### Вопрос пользователя:
        {question}

        ### Извлечённые фрагменты документации:
        {context}

        Проанализируй приведённые фрагменты и ответь на вопрос как можно точнее, строго на основе представленной информации. Не делай выводов вне содержания.

        - Отвечай только на основе вопроса, не уходи от темы вопроса.
        - Если ответ явно содержится в фрагментах — приведи его кратко (но не слишком кратко) и по существу.
        - Если в фрагментах есть ссылка на источник или указан источник — укажи эту ссылку или источник.
        - Если ответ можно предположить, но нет полной уверенности — укажи, что он вероятен, и поясни почему.
        - Если нужной информации нет — прямо скажи, что ответ отсутствует в фрагментах.

        Отвечай на русском языке. Не добавляй вводных или общих фраз.

        ### Ответ:
        """

        # 3. Генерируем ответ через локальную Mistral
        output = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["\n\n"]
        )

        answer = output.get('choices', [{}])[0].get('text', '').strip()
        return answer


if __name__ == "__main__":
    # Пример локального теста
    qe = QueryEngine(vectorstore_path=vectorstore_path, model_path=model_path, device=device)
    question = "какой Дисбаланс вендоров был у IT рынка по итогам 2022 года?"
    ans = qe.query(question)
    print("Ответ:\n", ans)
