# A-PRESTO-Embedding-Based-Framework-for-Rice-Non-Rice-Classification-and-Growth-Stage-Detection-
The pipeline runs in two stages:

Stage 1 — Rice vs. Non-Rice Classification
Multi-sensor satellite features (Sentinel-1 VV/VH radar, Sentinel-2 optical bands + NDVI, ERA5 climate variables, SRTM terrain, Dynamic World land cover) are extracted via Google Earth Engine for fixed sample points across Karnataka districts, spanning five Kharif seasons (2019–2023). These are passed through a fine-tuned PRESTO (Pretrained Remote Sensing Transformer) encoder to produce 128-dimensional embeddings capturing seasonal, multi-sensor crop signatures. An MLP classifier trained on these embeddings separates rice from non-rice pixels.

Stage 2 — Growth-Stage Analysis
Rice-classified pixels are further analyzed using monthly NDVI/LSWI/VV/VH temporal profiles across the Kharif season. Growth stages (Vegetative, Reproductive, Maturity/Senescence) are assigned via peak-detection on vegetation indices, refined using SAR backscatter trends (VV/VH) grounded in radar phenology literature, with a flooding-aware override from LSWI to correct early-season misclassification. An alternative K-means clustering approach groups temporal growth patterns into seasonal regimes.

What's in this repo
GEE extraction scripts (JavaScript) — district-level feature extraction with cloud-fallback handling, fixed-point sampling, and multi-year export
Data preparation — sequence building, NaN imputation, sentinel-value handling, CSV/parquet conversion
PRESTO embedding generation — multi-modal input construction and batched inference
Classifiers — MLP (PyTorch) and tree-based (RF/XGBoost) models for rice classification and growth-stage prediction
Growth-stage mapping — NDVI/LSWI/VV/VH-based phenological stage assignment, with diagnostic visualizations
Inference pipelines — scaled for both small fixed-point districts (Mandya, ~10K points) and large-scale regional rollout (Tumakuru, ~5.7M points)
Tech stack

Google Earth Engine · PyTorch · scikit-learn · XGBoost · pandas/NumPy · PRESTO (WorldCereal)
