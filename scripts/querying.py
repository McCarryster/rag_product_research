import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from llama_cpp import Llama
from rapidfuzz import fuzz

class QueryEngine:
    def __init__(
        self,
        model_path: str,
        vectorstore_path: str,
        embedding_model_name: str,
        n_ctx: int,
        top_k: int,
        device: str = "cpu",
        n_gpu_layers: int = 0,
    ):
        self.model_path = model_path
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.top_k = top_k

        # Load vectorstore
        self.vectorstore = None
        self._load_vectorstore()

        # Load the embedding model for query
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Load local LLM model
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

    # def _deduplicate_chunks(self, docs: list[Document]) -> list[str]:
    #     """Removes duplicate chunks (ignoring case and extra spaces)"""
    #     seen = set()
    #     unique_chunks = []
    #     for doc in docs:
    #         normalized = " ".join(doc.page_content.split()).lower()
    #         if normalized not in seen:
    #             seen.add(normalized)
    #             unique_chunks.append(doc.page_content.strip())
    #     return unique_chunks

    def _deduplicate_chunks(self, docs: list[Document]) -> list[str]:
        """Removes duplicate chunks (ignoring case and extra spaces)"""
        seen = set()
        unique_docs = []
        for doc in docs:
            normalized = " ".join(doc.page_content.split()).lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_docs.append(doc)

        # Fuzzy removal almost identical chunks
        final_chunks = []
        for doc in unique_docs:
            if all(fuzz.ratio(doc.page_content, existing) < 90 for existing in final_chunks):
                final_chunks.append(doc.page_content.strip())

        return final_chunks

    def query(
        self,
        query_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k_gen: int,
        typical_p: float,
        repeat_penalty: float
    ) -> str:
        # 1. Search for relevant docs
        query_embedding = self.embedding_model.embed_query(query_text)
        docs: list[Document] = self.vectorstore.similarity_search_by_vector(query_embedding, k=self.top_k)

        # 2. Clean duplicates
        unique_chunks = self._deduplicate_chunks(docs)

        # 3. Form context
        context = "\n---\n".join(unique_chunks)

        print("\n===== CONTEXT USED FOR THE ANSWER =====\n")
        print(context)
        print("\n==============================================\n")

        # 4. Form prompt on Russian
        prompt = f"""
Вопрос:
{query_text}

Контекст (только для справки, не повторяй дословно):
{context}

Инструкция по формированию ответа:
1. Проанализируй контекст и выдели только те факты, которые напрямую относятся к вопросу.
2. Удали все дубликаты фактов — если одно и то же число, дата или факт встречается несколько раз, используй его только один раз.
3. Объедини данные из разных частей контекста в одну логичную формулировку, без дословного копирования.
4. Для каждого уникального числа, даты или ключевого факта **укажи источник** (используй только те названия документов, статей или PDF, которые явно указаны в контексте). Источник пишется сразу после факта в скобках.
5. Не повторяй один и тот же источник для одинакового факта.
6. Излагай ответ развернуто, связно и логично, но без воды и повторов.
7. Не добавляй догадки, оценки или несвязанные данные, даже если они есть в контексте.
8. Не используй Markdown или HTML-разметку.
9. Ответ должен быть в виде цельного текста, а не списка.

- Перед тем как писать ответ, найди и удали из фактов любые повторяющиеся или почти одинаковые по смыслу данные, даже если они сформулированы по-разному.
- Если несколько фактов отличаются только формой записи (например, 3,5 млрд и 3.5 млрд), оставь только один.

Ответ:
"""

        # 5. Make inference
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_gen,
            typical_p=typical_p,
            repeat_penalty=repeat_penalty,
        )

        answer = output.get('choices', [{}])[0].get('text', '').strip()
        return answer
