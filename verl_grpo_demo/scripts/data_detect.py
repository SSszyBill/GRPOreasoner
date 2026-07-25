import os
from pathlib import Path
import pyarrow.parquet as pq

# 读取环境变量拼接路径
rl_root = os.environ["RL_ROOT"]
file_path = Path(rl_root) / "data/math500/test.parquet"

# 校验文件
if not file_path.exists():
    raise FileNotFoundError(f"目标parquet文件不存在: {file_path}")

# 读取parquet元信息
parquet_file = pq.ParquetFile(file_path)
meta = parquet_file.metadata

# 打印关键元数据
print("=== Parquet 文件元信息 ===")
# print(meta)
# print(f"总行数: {meta.num_rows}")
# print(f"总列数: {meta.num_columns}")
# print(f"文件schema:\n{meta.schema}")
# print(f"format_version: {meta.format_version}")
# print(f"serialized_size: {meta.serialized_size}")

# for i in range(meta.num_row_groups):
#     rg = meta.row_group(i)
#     print(f"  row_group[{i}]: {rg.num_rows:>7,} rows, "
#           f"{rg.total_byte_size / 1024**2:>6.1f} MB")

# for i in range(meta.num_columns):
#     col = meta.schema.column(i)
#     print(f"  column[{i}]: {col.name} ({col.physical_type})")


print("\n[sample first row]")
first_batch = next(parquet_file.iter_batches(batch_size=1))
sample_dict = first_batch.to_pylist()[0]
for k, v in sample_dict.items():
    preview = str(v)
    if len(preview) > 120:
        preview = preview[:120] + "..."
    print(f"  {k}: {preview}")