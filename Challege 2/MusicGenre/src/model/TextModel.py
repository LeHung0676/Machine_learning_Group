"""Model văn bản: TF-IDF n-gram ký tự trên "nghệ sĩ || tên bài" + hồi quy logistic.

Model này KHÔNG dùng thuộc tính âm thanh nào. Riêng lẻ nó yếu, nhưng nó nhìn
dữ liệu từ một góc hoàn toàn khác các model còn lại, nên có ích khi kết hợp.
"""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model.base import BaseModel


class TextModel(BaseModel):
    name = "text"

    def __init__(self, cfg: dict, pre):
        self.cfg, self.pre = cfg, pre
        self.seed = cfg["project"]["seed"]
        p = self.params = dict(cfg["models"].get(self.name, {}))
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(p.get("ngram_min", 2), p.get("ngram_max", 4)),
            min_df=p.get("min_df", 2), sublinear_tf=True,
        )
        self.model = Pipeline([
            ("enc", ColumnTransformer([("tfidf", tfidf, "pair")])),
            ("clf", LogisticRegression(C=p.get("C", 5.0), max_iter=2000)),
        ])
