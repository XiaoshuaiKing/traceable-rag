"""Load only the project's own embedding model, without global model caches."""
from . import config

def load_embedding_model(model_name=None):
    from sentence_transformers import SentenceTransformer
    path=config.BGE_MODEL_DIR
    if not (path/'config.json').is_file():
        raise FileNotFoundError(f'Embedding model missing: {path}. Run python download_embedding.py')
    if model_name is not None and model_name!=config.EMBED_MODEL_BGE:
        raise ValueError('Requested embedding identity differs from the project index profile')
    return SentenceTransformer(str(path),local_files_only=True)
