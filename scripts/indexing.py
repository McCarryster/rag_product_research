import os
from typing import Dict, List
import gc
import torch
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from extract_text import extract_texts_from_folder
from config import *

class VectorStoreBuilder:
    def __init__(
        self,
        embedding_model_name: str,
        vectorstore_path: str,
        chunk_size: int,
        chunk_overlap: int,
        device: str = "cpu"
    ):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.vectorstore_path = vectorstore_path

        self.embedding_model = None
        self.splitter = None
        self.vectorstore = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self.embedding_model

    def _get_splitter(self):
        if self.splitter is None:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        return self.splitter

    def load_vectorstore(self):
        index_file = os.path.join(self.vectorstore_path, "index.faiss")
        pkl_file = os.path.join(self.vectorstore_path, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self._get_embedding_model(),
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = None

    def get_existing_uids(self) -> set:
        existing_uids = set()
        if self.vectorstore and self.vectorstore.docstore:
            for doc_id, doc in self.vectorstore.docstore._dict.items():
                if doc and doc.metadata:
                    uid = doc.metadata.get("uid")
                    if uid is not None:
                        existing_uids.add(uid)
        return existing_uids

    def _prepare_documents(self, texts: Dict[str, str]) -> List[Document]:
        """
        Converts a dictionary {filename: text} into a list of Document objects with uid=filename.
        """
        docs = []
        for fname, text in texts.items():
            docs.append(Document(page_content=text, metadata={"uid": fname}))
        return docs

    def filter_new_documents(self, docs: List[Document]) -> List[Document]:
        existing_uids = self.get_existing_uids()
        new_docs = [doc for doc in docs if doc.metadata.get("uid") not in existing_uids]
        return new_docs

    def add_texts(self, texts: Dict[str, str]):
        """
        Adds new documents from the dictionary {filename: text} to the vectorstore.
        Creates the vectorstore if it does not exist yet.
        """
        self.load_vectorstore()
        docs = self._prepare_documents(texts)
        new_docs = self.filter_new_documents(docs)
        if not new_docs:
            print("[INFO] No new documents to add.")
            return

        splitter = self._get_splitter()
        split_docs = splitter.split_documents(new_docs)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, self._get_embedding_model())
        else:
            self.vectorstore.add_documents(split_docs)

        self.save_vectorstore()
        print(f"[INFO] Added {len(split_docs)} chunks out of {len(new_docs)} new docs.")

    def save_vectorstore(self):
        if self.vectorstore is not None:
            os.makedirs(self.vectorstore_path, exist_ok=True)
            self.vectorstore.save_local(self.vectorstore_path)

    def cleanup(self):
        # If gonna depoy to clean memory
        self.vectorstore = None
        self.embedding_model = None
        self.splitter = None
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    texts = extract_texts_from_folder(data_dir)
    builder = VectorStoreBuilder(
        embedding_model_name=embedding_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        vectorstore_path=vectorstore_path
    )
    builder.add_texts(texts)
    builder.cleanup()