import asyncio

from chromadb import HttpClient, PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from team_a_dspy.services.config import settings
from team_a_dspy.services.es_client import ESClient


class ChromaClient:
    """
    Client for interacting with ChromaDB vector store.
    This is used to store and retrieve vector embeddings for gdelt metadata fields.
    """
    def __init__(self, dev: bool = False) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.chroma_embedding_model)
        self.collection_name = settings.chroma_collection_name
        if dev:
            client = PersistentClient(path=settings.chroma_persistent_path)
        else:
            client = HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )

        self.vectorstore: Chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=client
        )

    async def initialize_data(self, es_client: ESClient):
        """
        Call this method immediately after instantiating the class to load initial data.
        """
        if es_client and self.vectorstore._collection.count() == 0:
            await self.add_documents(es_client)

    async def add_documents(self, es_client: ESClient):
        """
        Add a list of langchain.schema.Document to Chroma and persist.
        """
        pass

    def similarity_search(self, query: str, k: int = 6) -> list[Document]:
        """
        Perform a similarity search against the vector store.
        Returns a list of langchain.schema.Document.
        """
        return self.vectorstore.similarity_search(query, k=k)
