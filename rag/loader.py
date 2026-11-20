"""Strict document loading and one consistent BGE parent/child rebuild path."""
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from .store import Doc, open_child_store


def normalize_metadata(docs):
    for doc in docs:
        source = Path(doc.metadata.get('source', 'unknown'))
        try:
            relative = source.resolve().relative_to(DOCS_DIR.resolve()).as_posix()
        except ValueError:
            relative = source.name
        doc.metadata.update(source_file=source.name, source_path=relative,
                            source_type=source.suffix.lower().lstrip('.'))
    return docs


def load_documents():
    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f'Document directory not found: {DOCS_DIR}')
    documents = []
    for file in sorted(DOCS_DIR.rglob('*')):
        if not file.is_file() or file.name.startswith('~$'):
            continue
        suffix = file.suffix.lower()
        if suffix in ('.txt', '.md'):
            def read():
                return [Doc(file.read_text(encoding='utf-8'), {'source': str(file)})]
        elif suffix == '.pdf':
            def read():
                from pypdf import PdfReader
                reader = PdfReader(file)
                return [Doc(page.extract_text() or '', {'source': str(file), 'page': i}) for i, page in enumerate(reader.pages)]
        elif suffix == '.docx':
            def read():
                with zipfile.ZipFile(file) as archive:
                    tree = ET.fromstring(archive.read('word/document.xml'))
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                text = '\n\n'.join(''.join(t.text or '' for t in p.findall('.//w:t', ns)) for p in tree.findall('.//w:body//w:p', ns))
                return [Doc(text, {'source': str(file)})]
        else:
            continue
        try:
            documents.extend(normalize_metadata(read()))
        except Exception as exc:
            # Never activate a partial corpus after a parser failure.
            raise RuntimeError(f'Failed to load {file.name}; active index unchanged') from exc
    if not documents:
        raise ValueError('No supported documents; active index unchanged')
    return documents


def prepare_chunks(docs):
    from .index import digest
    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError('Chunk overlap must be smaller than chunk size')
    chunks = []
    occurrences = {}
    split_docs = []
    for doc in docs:
        text, start = doc.page_content, 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            if end < len(text):
                breaks = list(re.finditer(r'\n\n|\n|[。！？]', text[start + CHUNK_SIZE // 2:end]))
                if breaks:
                    end = start + CHUNK_SIZE // 2 + breaks[-1].end()
            split_docs.append(Doc(text[start:end], dict(doc.metadata)))
            if end == len(text):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
    for doc in split_docs:
        if not doc.page_content.strip():
            continue
        meta = doc.metadata
        identity = {'text': doc.page_content, 'source_path': meta['source_path'],
                    'page': meta.get('page', -1)}
        key = digest(identity)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        chunks.append({**identity, 'source': meta['source_file'],
                       'chunk_id': key[:24] + f'_{occurrence}'})
    return chunks


def build_vector_store(docs):
    from .index import build_from_chunks
    build_from_chunks(prepare_chunks(docs))
    return open_child_store()


def init_vector_store(rebuild=False):
    if rebuild:
        return build_vector_store(load_documents())
    try:
        return open_child_store()
    except Exception as exc:
        # Incompatible embedding/distance profiles must never be silently loaded.
        raise RuntimeError('Index unavailable. Run python -m rag.index (documents) or '
                           'python -m rag.index --from-snapshot (frozen benchmark).') from exc
