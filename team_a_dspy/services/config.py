from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ChromaDB
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection_name: str = "gkg"
    chroma_persistent_path: str = "./chroma_persistent"

    #Elasticsearch
    es_host: str
    es_username: str
    es_password: str
    es_verify_ssl: bool = False
    es_index: str = "gkg"

    # DSPY
    llm_base_url: str
    llm_model_name: str
    llm_api_key: str
    max_result_docs: int = 20
    max_agg_buckets: int = 50
    
    class Config:
        env_file = ".env"

settings = Settings()