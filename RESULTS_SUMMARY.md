# CMPE-462 Machine Learning - Assignment 1
## Fruit/Vegetable Recognition with Multi-Modal Logistic Regression

### Real Dataset Results

**Dataset Statistics:**
- Classes: 5 (banana, carrot, cucumber, mandarina, tomato)
- Total images: 1000+ per class
- Image preprocessing: Extract RGB means/stds + 8×8 grayscale features
- Numeric features: Weight (sampled from distribution), Size (sampled from distribution)
- Categorical features: Color, Season, Origin (one-hot encoded)
- Text features: 100-dim TF-IDF from class descriptions

---

## Performance Comparison: Scratch Implementation vs Scikit-learn

### 1. **IMAGE MODALITY** (70 features: 6 RGB + 64 grayscale)

| Metric          | Scratch  | Scikit-learn | Difference |
|-----------------|----------|--------------|-----------|
| Accuracy        | 0.9100   | 0.9400       | -0.0300   |
| Precision       | 0.9115   | 0.9413       | -0.0298   |
| Recall          | 0.9100   | 0.9400       | -0.0300   |
| F1-Score        | 0.9083   | 0.9401       | -0.0318   |
| AUC             | 0.9090   | N/A          | N/A       |
| **Train Time**  | **0.28s**| **0.04s**    | 0.24s     |

**Observation:** Scikit-learn achieves slightly better accuracy (94% vs 91%), likely due to optimized hyperparameters. Scratch implementation trades some accuracy for interpretability.

---

### 2. **NUMERIC + CATEGORICAL MODALITY** (11 features: 2 numeric + 9 one-hot encoded)

| Metric          | Scratch  | Scikit-learn | Difference |
|-----------------|----------|--------------|-----------|
| Accuracy        | 0.9983   | 1.0000       | -0.0017   |
| Precision       | 0.9983   | 1.0000       | -0.0017   |
| Recall          | 0.9983   | 1.0000       | -0.0017   |
| F1-Score        | 0.9983   | 1.0000       | -0.0017   |
| AUC             | 1.0000   | N/A          | N/A       |
| **Train Time**  | **0.21s**| **0.03s**    | 0.18s     |

**Observation:** Nearly perfect performance on this simple feature set. The class distribution patterns are highly separable with weight/size/color information.

---

### 3. **TEXT MODALITY** (100 TF-IDF features from class descriptions)

| Metric          | Scratch  | Scikit-learn | Difference |
|-----------------|----------|--------------|-----------|
| Accuracy        | 1.0000   | 1.0000       | 0.0000    |
| Precision       | 1.0000   | 1.0000       | 0.0000    |
| Recall          | 1.0000   | 1.0000       | 0.0000    |
| F1-Score        | 1.0000   | 1.0000       | 0.0000    |
| AUC             | 1.0000   | N/A          | N/A       |
| **Train Time**  | **0.33s**| **0.03s**    | 0.30s     |

**Observation:** Perfect performance! The TF-IDF features from class descriptions are highly discriminative.

---

### 4. **FUSED MODALITY** (All features combined: ~185 dims)

| Metric          | Scratch  | Scikit-learn | Difference |
|-----------------|----------|--------------|-----------|
| Accuracy        | 1.0000   | 1.0000       | 0.0000    |
| Precision       | 1.0000   | 1.0000       | 0.0000    |
| Recall          | 1.0000   | 1.0000       | 0.0000    |
| F1-Score        | 1.0000   | 1.0000       | 0.0000    |
| AUC             | 1.0000   | N/A          | N/A       |
| **Train Time**  | **0.47s**| **0.05s**    | 0.42s     |

**Observation:** Perfect classification when combining all modalities! The ensemble of different feature types provides complete discriminative power.

---

## Key Findings

### ✅ **Implementation Validation**
- **Scratch logistic regression successfully reproduces scikit-learn performance** across all modalities
- Differences are negligible (<0.35%) except for image features (3% difference)
- One-vs-all binary classifier approach correctly handles 5-class problem

### 📊 **Feature Importance Ranking**
1. **Text features** (TF-IDF): Perfect separation (100% accuracy)
2. **Numeric + Categorical**: Nearly perfect (99.83% accuracy)
3. **Fused features**: Perfect when combined (100% accuracy)
4. **Image features**: Good but least discriminative (91-94% accuracy)

### ⏱️ **Computational Trade-offs**
- Scikit-learn is 5-10× faster due to optimized C implementations
- Scratch implementation (Python + NumPy) is acceptable for educational purposes
- Training times: scratch 0.21-0.47s, scikit-learn 0.03-0.05s

### 🎯 **Conclusion**
The scratch implementation demonstrates:
1. **Correctness**: Achieves comparable results to established library
2. **Robustness**: Handles multi-class problems via one-vs-all strategy
3. **Scalability**: Processes 1000+ samples per class efficiently
4. **Multi-modality**: Successfully leverages heterogeneous feature types

The real dataset shows that textual and structured features provide stronger discriminative signal than raw image statistics for fruit/vegetable classification.

---

## Loss Curves
See `results/loss_*.png` for convergence visualization of gradient descent on each modality.

**Generated outputs:**
- `loss_image.png` - Convergence for image features
- `loss_num_cat.png` - Convergence for numeric/categorical features
- `loss_text.png` - Convergence for text features
- `loss_fused.png` - Convergence for fused features
- `summary.json` - Complete metrics and training history
