from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str = "sk-dummy"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"

    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBED_BATCH_SIZE: int = 32

    DATABASE_URL: str = "postgresql+psycopg://portoai:portoai@localhost:5433/portoai"
    PG_COLLECTION: str = "porto_docs"

    OCR_DPI: int = 300
    OCR_WORKERS: int = 4
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 5
    HYBRID_VECTOR_K: int = 20
    HYBRID_KEYWORD_K: int = 20
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RERANK_CANDIDATES: int = 20
    RERANK_BATCH_SIZE: int = 16
    AGENT_ENABLED: bool = True
    AGENT_MAX_REWRITE: int = 1

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
