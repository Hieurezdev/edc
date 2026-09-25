from datasets import load_dataset

base = "hf://datasets/Babelscape/SREDFM@~parquet/vi"

ds = load_dataset(
    "parquet",
    data_files={
        "train": f"{base}/train/*.parquet",
        "validation": f"{base}/validation/*.parquet",
        "test": f"{base}/test/*.parquet",
    },
)
ds.save_to_disk("/data/GraphJudge/data/sredfm_vi")
print(ds)