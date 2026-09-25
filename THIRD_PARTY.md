# Vendored source code

The following projects are included as ordinary source files so the Vietnamese pipeline can run from this repository. Their original license files are retained in their respective directories.

| Directory | Upstream | Base revision | License | Local additions |
|---|---|---|---|---|
| `GraphJudge/` | [Hieurezdev/GraphJudge](https://github.com/Hieurezdev/GraphJudge) | `7c1fb8c367480d36bd0e14203594ebba9e1d4967` | [MIT](GraphJudge/LICENSE) | `graph_judger/verify_triples.py` for source-grounded Vietnamese triple verification |
| `visolex/` | [HaDung2002/visolex](https://github.com/HaDung2002/visolex) | `53c62c728eb07610fafdde25df86f0340069ec19` | [MIT](visolex/LICENSE) | Local checkpoint-loading compatibility change in `normalizer/model_construction/bartpho.py` |

Large GraphJudge datasets, crawled books, generated book copies, and model checkpoints are excluded from Git. The source scripts used to prepare GraphJudge datasets remain in `GraphJudge/datasets/`.
