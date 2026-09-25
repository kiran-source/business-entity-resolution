# Entity Resolution Validation Error Analysis

- **True Positive Pairs**: 5188
- **False Positive Pairs**: 45
- **False Negative Pairs**: 1075
- **Singleton False Positives**: 4
- **Non-Singleton False Negatives**: 53

### Error Pattern Insights
1. **High Name Similarity False Positives**: Different branches/franchises sharing name tokens.
2. **Transliteration & Typo False Negatives**: Indic script variations without sufficient token overlap.
3. **Empty Address Uncertainty**: When target address is missing, match confidence depends entirely on name specificity.
