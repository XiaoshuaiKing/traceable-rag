"""Project-local paths and retrieval profile; no dependency on other projects."""
import os
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
def project_path(key,default):
    path=Path(os.getenv(key,str(default)))
    return (PROJECT_ROOT/path).resolve() if not path.is_absolute() else path.resolve()

DATA_DIR=project_path('RAG_DATA_DIR','data')
DOCS_DIR=project_path('RAG_DOCS_DIR',DATA_DIR/'docs')
CHROMA_DIR=project_path('RAG_CHROMA_DIR',DATA_DIR/'index')
CHUNKS_JSON=project_path('RAG_CHUNKS_JSON',DATA_DIR/'chunks.json')
EVAL_RESULTS_DIR=PROJECT_ROOT/'eval/results'
EMBED_MODEL_BGE='BAAI/bge-small-zh-v1.5'
BGE_MODEL_DIR=project_path('RAG_MODEL_DIR','models/bge-small-zh-v1.5')
INDEX_MANIFEST=CHROMA_DIR/'index_manifest.json'
INDEX_SCHEMA=1
CHILD_MIN_LEN=25
CHILD_MAX_LEN=120
CHILD_RECALL_POOL=60
GATE_SLOPE=1.25
GATE_THRESHOLD=.4
# Raw Chroma L2 distance, independent of distance - GATE_SLOPE * coverage.
# No vector-only rejection threshold has been calibrated for this index.
VECTOR_DISTANCE_THRESHOLD=None
FUSION_WEIGHT=.45
CHUNK_SIZE=500
CHUNK_OVERLAP=50
