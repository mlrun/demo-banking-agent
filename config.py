from pydantic import BaseModel


class MainConfig(BaseModel):
    source_url: str = "store://datasets/banking-agent/churn#0:latest"
    label_column: str = "churn"
    allow_validation_failure: bool = True
    test_size: float = 0.2
    model_name: str = "churn_model"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    text_column: str = "chat_log"
    sentiment_column: str = "sentiment_label"
    ordinal_columns: list = ["international_plan", "voice_mail_plan"]
    drop_columns: list = ["area_code", "chat_log", "state"]

workflow_configs = {"main": MainConfig()}
