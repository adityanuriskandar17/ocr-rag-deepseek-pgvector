from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str = "sk-dummy"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    DATABASE_URL: str = "postgresql+psycopg://portoai:portoai@localhost:5432/portoai"
    PG_COLLECTION: str = "porto_docs"

    OCR_DPI: int = 300
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
