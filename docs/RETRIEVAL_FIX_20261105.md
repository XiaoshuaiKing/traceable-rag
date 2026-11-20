# 检索实现修复与回归记录（2026-11-05）

## 修复内容

### A：纯向量距离与混合闸门分离

`ChildChromaStore` 及其 manifest 加载路径不再把混合闸门的 `max_score` 写入原始距离阈值。新增 `rag/config.py` 的 `VECTOR_DISTANCE_THRESHOLD`，默认 `None`；`retriever_vec.search_documents()` 在阈值为 `None` 时返回最近邻，数值阈值只比较原始 Chroma L2 距离且包含等值边界。

混合闸门仍按 `nearest_distance - cover_slope * coverage > max_score` 判断；其 `0.4` 不能视为纯距离校准结果。旧模型的 `13.5` 回退也已移除。需要启用纯向量过滤时，应针对当前模型和索引在开发集独立校准，再设置 `VECTOR_DISTANCE_THRESHOLD` 或实例的 `distance_threshold`。默认纯向量路径不具有已校准的库外拒答保证。

### B：BM25 正分候选

`BM25.rank()` 只保留分数大于0的下标再稳定排序。`_bm_rank()` 在此基础上截取 TopK，所以零分文档不会占名次或获得 RRF 加分；无匹配、空串、空白或单字查询没有 bigram 时返回空列表。向量候选仍可进入融合。

## 回归测试

新增 `tests/test_retrieval_boundaries.py` 的14项测试，涵盖默认/显式/零值阈值、等值边界、空候选、manifest加载时阈值隔离、BM25正分排序与同分稳定性、无词面命中及RRF候选集合。新增测试在修复前复现问题；修复后连同原15项回归共29项全部通过。

```powershell
Set-Location E:\RAG_project
& D:\Anaconda\envs\dl\python.exe -B -m unittest discover -s tests -v
```

## 真实纯向量入口核验

查询：`Agent 有哪四个核心部件？`。真实前五名原始距离约为 0.4241、0.8227、0.8413、0.8475、0.8505。

修复前所有距离大于误用的0.4，纯向量问答入口会把这批候选全部过滤；修复后默认返回5条证据。将实例原始距离阈值显式设为最近邻距离时返回1条，验证等值保留与超阈值过滤。这个样例证明过滤语义修复，不等于五条候选都相关或生成答案已正确。

## 固定参数回归评测

语料、题集和混合参数与原结果保持一致；未调参、未重建索引、未调用生成接口。原开发集和已分析诊断集按回归用途使用，不宣称新的独立盲测。

| 数据集 | 样本数 | Vector R@5 | BM25 R@5 | Hybrid R@5 | 闸门 FRR / FAR |
|---|---:|---:|---:|---:|---:|
| development | 400 | 79.3% | 82.3% | 83.8% | 7.5% / 5.5% |
| challenge | 64 | 78.1% | 90.6% | 84.4% | 62.5% / 28.1% |

上述R@5和闸门FRR/FAR与修复前相同；这是实现正确性修复，不据此宣称召回提升。现有发布评测的vector排序直接调用未过滤向量召回，本来就不经过 `retriever_vec.search_documents()`，所以它的vector R@5不能反映修复前问答入口被0.4误过滤的问题，需结合上面的真实入口核验。

- development：Top5顺序/集合变化题数，vector=0、BM25=57、hybrid=82；Hybrid MRR@5 0.684933 → 0.685774。
- challenge：Top5顺序/集合变化题数，vector=0、BM25=3、hybrid=7；Hybrid MRR@5 0.675521 → 0.675521。

诊断集误拒与误接受问题依然存在，不属于这两处实现修复已经解决的能力。

## 结果文件

- [修复后 development 逐题结果](../eval/results/release_development_retrieval_fix_20261105.json)
- [修复后 challenge 逐题结果](../eval/results/release_challenge_retrieval_fix_20261105.json)
- 原 `release_development.json`、`release_challenge.json` 保留作历史对照。
