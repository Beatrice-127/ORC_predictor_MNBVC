## ocr_predict.py 使用说明

`ocr_predict.py` 是一个批量推理脚本，用来遍历大量 PDF，抽取与 `construct_data.ipynb` 相同的版面特征，并利用 `xgb_ocr_classifier/xgb_classifier.ubj` 模型估计每个文件是否需要 OCR。脚本会递归搜索输入目录、对每份文档按页采样计算特征、输出包含概率和诊断信息的 CSV，便于在大规模数据集上筛选待 OCR 的文件。

---

### 1. 环境与准备

- Python 3.9+，并已安装脚本顶部导入的依赖（`pymupdf`, `numpy`, `pandas`, `xgboost` 等）。  
- 模型文件：`xgb_ocr_classifier/xgb_classifier.ubj`（默认路径为仓库同名目录，可通过 `--model-path` 覆盖）。  
- 确保输入目录/文件可读，输出路径可写。

建议先在虚拟环境或容器中执行一次依赖安装，例如：

```bash
uv pip install -r requirements.txt  # 或手动安装 pymupdf numpy pandas xgboost
```

---

### 2. 基本用法

```bash
python ocr_predict.py <pdf_or_dir> [更多目录或文件] \
  --model-path xgb_ocr_classifier/xgb_classifier.ubj \
  --output my_predictions.csv
```

脚本会：

1. 对所有输入路径去重，递归查找 `.pdf` 文件。  
2. 使用 `PDFFeatureExtractor` 在每份文档中采样页面（默认 8 页、1 个 chunk），计算文档级与页级特征。  
3. 通过 `XGBClassifier` 输出需要 OCR 的概率，并按 `--threshold`（默认 0.5）打标签。  
4. 将推理结果写入 `--output` 指定的 CSV。失败记录会写到 `--failures-output` 或默认的 `<output>.failures.csv`。

---

### 3. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `inputs` | 一个或多个 PDF 文件/目录；目录会递归搜索所有 `.pdf`。 |
| `--model-path` | 模型文件路径，默认 `xgb_ocr_classifier/xgb_classifier.ubj`。 |
| `--output` | 预测结果 CSV，默认 `ocr_predictions.csv`。 |
| `--failures-output` | 失败样本记录 CSV（可选；默认写在 `<output>.failures.csv`）。 |
| `--sample-pages` | 单个 chunk 内采样页数，默认 8。 |
| `--num-chunks` | 采样 chunk 数；`-1` 时会覆盖到所有页，否则默认 1。 |
| `--threshold` | 将概率转换为是否需要 OCR 的阈值，默认 0.5。 |
| `--seed` | 控制随机采样/重采样的随机种子，默认 13。 |
| `--max-files` | 限制最多处理多少 PDF，用于抽样或调试。 |
| `--log-every` | 每处理多少文件打印一次进度日志，默认 25。 |
| `--quiet` | 只输出警告/错误日志。 |

---

### 4. 输出内容

`--output` CSV 包含下列字段：

- `path`：PDF 绝对路径。  
- `ocr_probability`：模型估计需要 OCR 的概率。  
- `needs_ocr`：是否超过 `--threshold`。  
- `is_form`、`garbled_text_ratio`、`num_pages_successfully_sampled`：诊断特征，方便按类型过滤。  
- `num_pages`、`is_encrypted`、`needs_password`：文档元数据。

若某些文件损坏或解密失败，会记录到失败 CSV 中，字段为 `path` 和 `error`。

---

### 5. 示例

```bash
# 对 le/ 与 ge/ 两个目录批量推理，将结果输出到 batch_preds.csv
python ocr_predict.py le ge \
  --model-path xgb_ocr_classifier/xgb_classifier.ubj \
  --output batch_preds.csv \
  --threshold 0.55 \
  --sample-pages 8 \
  --num-chunks 1

# 调试模式：只扫描 finepdfs 目录中的前 20 个文件
python ocr_predict.py finepdfs --max-files 20 --quiet
```

---

### 6. 建议工作流

1. 先用 `--max-files` 在少量样本上测试，确认依赖和模型加载正常。  
2. 根据 OCR 预算或实际需求调整 `--threshold`；可以通过人工标注的一小批样本来校准。  
3. 对输出 CSV 分组/筛选，例如按 `needs_ocr` 或 `garbled_text_ratio` 过滤，再交给 OCR 管线。  
4. 保留失败 CSV，针对加密或损坏文件制定单独的处理策略。

如需定制采样策略或新增特征，可以直接编辑 `ocr_predict.py` 中 `PDFFeatureExtractor` 或 `flatten_per_page_features` 的实现。

