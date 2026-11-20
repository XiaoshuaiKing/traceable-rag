"""Read the active parent/child index and aggregate nearest child distances."""
import json
from pathlib import Path
from . import config
from .embedding import load_embedding_model

class Doc:
    __slots__=('page_content','metadata')
    def __init__(self,page_content,metadata):
        self.page_content=page_content
        self.metadata=metadata

class ChildChromaStore:
    def __init__(self,child_collection,model,pool=config.CHILD_RECALL_POOL,parent_collection=None):
        self._collection=child_collection
        self.model=model
        self.pool=pool
        self._parent=parent_collection
        self._corpus=None
        self.profile={'w_vec':config.FUSION_WEIGHT,'cover_slope':config.GATE_SLOPE,'max_score':config.GATE_THRESHOLD}
        self.version='unversioned'
        self.distance_threshold=config.VECTOR_DISTANCE_THRESHOLD

    @property
    def corpus(self):
        if self._corpus is None:
            raw=self._parent.get(include=['documents','metadatas'])
            pairs=sorted(zip(raw['metadatas'],raw['documents']),key=lambda row:row[0]['parent_id'])
            self._corpus=([doc for _,doc in pairs],[meta for meta,_ in pairs])
        return self._corpus

    def similarity_search_with_score(self,query,k=5):
        if k<=0:raise ValueError('k must be positive')
        n=min(self.pool,self._collection.count())
        if n==0:return []
        vector=self.model.encode([query],normalize_embeddings=True,show_progress_bar=False)[0].tolist()
        raw=self._collection.query(query_embeddings=[vector],n_results=n,include=['metadatas','distances'])
        best={}
        for meta,distance in zip(raw['metadatas'][0],raw['distances'][0]):
            pid=meta['parent_id']
            if pid not in best or distance<best[pid][0]:best[pid]=(distance,meta)
        ranked=sorted(best.items(),key=lambda item:item[1][0])[:k]
        return [(Doc(meta['parent_text'],{key:value for key,value in meta.items() if key!='parent_text'}),distance)
                for _,(distance,meta) in ranked]

def open_child_store(chroma_dir=None,model_name=None):
    import chromadb
    directory=Path(chroma_dir) if chroma_dir is not None else config.CHROMA_DIR
    manifest_path=directory/'index_manifest.json'
    if not manifest_path.is_file():raise FileNotFoundError('Active index missing; run python -m rag.index --from-snapshot')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    spec=manifest['spec']
    if spec['model']!=config.EMBED_MODEL_BGE or spec['distance']!='l2' or not spec['normalize_embeddings']:
        raise ValueError('Index model/distance profile does not match project configuration')
    client=chromadb.PersistentClient(path=str(directory))
    parent=client.get_collection(manifest['parent_collection'],embedding_function=None)
    child=client.get_collection(manifest['child_collection'],embedding_function=None)
    if parent.count()!=manifest['parents'] or child.count()!=manifest['children'] or not parent.count() or not child.count():
        raise ValueError('Active index is incomplete')
    store=ChildChromaStore(child,load_embedding_model(model_name),parent_collection=parent)
    store.profile={key:manifest['profile'][key] for key in ['w_vec','cover_slope','max_score']}
    store.version=manifest['version']
    return store
