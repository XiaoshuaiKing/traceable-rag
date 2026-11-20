"""Install the project's embedding model when preparing a fresh checkout."""
import json
from rag.config import BGE_MODEL_DIR,EMBED_MODEL_BGE
if (BGE_MODEL_DIR/'config.json').is_file() and (BGE_MODEL_DIR/'model.safetensors').is_file():
    print('Using project embedding model:',BGE_MODEL_DIR)
else:
    from huggingface_hub import HfApi,snapshot_download
    revision=HfApi().model_info(EMBED_MODEL_BGE).sha
    snapshot_download(EMBED_MODEL_BGE,revision=revision,local_dir=BGE_MODEL_DIR,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.md','LICENSE'])
    (BGE_MODEL_DIR/'source.json').write_text(json.dumps({'repo':EMBED_MODEL_BGE,'revision':revision},indent=2),encoding='utf-8')
