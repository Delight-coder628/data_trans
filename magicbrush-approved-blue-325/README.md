# MagicBrush 蓝区审核数据传输包（325 对）

此目录存放拆分后的 `magicbrush-approved-blue-325.zip`，共 12 个分卷，单卷最多 20 MB，低于 GitHub 的 100 MB 单文件限制。蓝区审核结果：原候选 390 对，按填写的 `reason` 排除 62 对，另有 3 对继续待复核；本包仅包含其余 325 对。

在黄区获取此目录的全部文件后，从本目录运行：

```bash
python assemble.py
unzip magicbrush-approved-blue-325.zip -d magicbrush-approved-blue-325
cd magicbrush-approved-blue-325
python finalize_magicbrush_review.py --package-root . --decisions review_decisions.csv --output magicbrush.approved.jsonl
```

`assemble.py` 会检查每个分卷的大小、重组文件的总大小及 ZIP CRC。`finalize_magicbrush_review.py` 会再次验证图片并生成黄区本地绝对路径的 OmniGen2 JSONL。不要直接把分卷或候选清单入训，也不要混入未经审核的黄区样本。MagicBrush ID 不能交给 PIE 专用的 `merge_validate_training_data.py` 处理。

数据来源：MagicBrush train，许可 [CC BY 4.0](https://huggingface.co/datasets/osunlp/MagicBrush)。
