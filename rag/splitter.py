# -*- coding: utf-8 -*-
"""句子切分：把整块文本拆成检索用的子块。

中文没有空格分词，按句末标点切最稳；标点不全的段落（表格行、条目）
会用换行兜底。过短的片段并进下一句，过长的再按逗号二次切。
"""


def split_sentences(text, min_len=25, max_len=120):
    """返回子块列表。min_len 决定聚合到多长才断开，max_len 是硬上限。"""
    import re

    parts = re.split(r"(?<=[。！？；\n])", text)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf += p
        if len(buf) < min_len:
            continue
        if len(buf) > max_len:
            # 长句按逗号再切，避免子块过长把语义又摊薄
            cur = ""
            for s in re.split(r"(?<=[，、])", buf):
                if len(cur) + len(s) > max_len and cur:
                    out.append(cur)
                    cur = s
                else:
                    cur += s
            if cur:
                out.append(cur)
        else:
            out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return [part[i:i + max_len] for part in out for i in range(0, len(part), max_len)]
