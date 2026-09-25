# Entity Resolution Data Profiling Summary Report

## Ground Truth Distribution

- **Total S1 Entities**: 2,206,821
- **Zero Matches (Singletons)**: 123,247 (5.58%)
- **Exactly 1 Match**: 119,157 (5.4%)
- **Multiple Matches**: 1,964,417 (89.02%)
- **Total Positive Pairs**: 7,638,365
- **Average Matches per S1**: 3.461
- **Max Matches for single S1**: 11

## Source File Completeness and Distributions

| File | Total Records | Missing Names | Missing Addr (%) | Countries |
| --- | --- | --- | --- | --- |
| train_source1.tsv | 2,206,821 | 0 | 0.0% | US: 1,323,633, India: 883,188 |
| train_source2.tsv | 5,034,616 | 0 | 3.3561% | India: 2,017,799, US: 3,016,817 |
| train_source3.tsv | 5,285,603 | 0 | 3.3282% | US: 3,170,056, India: 2,115,547 |
| test_source1.tsv | 1,732,544 | 0 | 0.0% | US: 663,106, France: 259,452, India: 809,986 |
| test_source2.tsv | 4,887,273 | 0 | 2.6479% | India: 2,312,565, France: 703,378, US: 1,871,330 |
| test_source3.tsv | 5,082,316 | 0 | 2.6779% | India: 2,405,000, France: 731,615, US: 1,945,701 |
