# Dataset Specifications

## Overview
This directory is designated for the 7 official Business Entity Resolution Challenge TSV files:

### Training Data (`dataset/train/`)
- `train_source1.tsv`: Reference deduplicated Source 1 business records (~2.2M records).
- `train_source2.tsv`: Noisy Source 2 records (~5.0M records).
- `train_source3.tsv`: Noisy Source 3 records (~5.3M records).
- `train_ground_truth.tsv`: Source 1 entity mappings to matched Source 2 and Source 3 entities.

### Test Data (`dataset/test/`)
- `test_source1.tsv`: Source 1 test records (~1.73M records).
- `test_source2.tsv`: Source 2 test records (~4.89M records).
- `test_source3.tsv`: Source 3 test records (~5.08M records).

## File Format & Parsing
- All files are strictly tab-separated (`sep="\t"`) with UTF-8 encoding.
- Columns for source files: `entity_id`, `business_name`, `business_address`, `country`.
- Columns for ground truth: `source1_entity_id`, `matched_entity_ids`.
- Comma-separated ID lists in `matched_entity_ids`. Empty values represent singletons (entities with zero matches).

## Open-Set Country Generalization
- The training data covers `US` and `India`.
- The test set additionally introduces `France`.
- Entity resolution is strictly partitioned by country; businesses never match across international borders.
