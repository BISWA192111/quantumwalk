# Maritime Network Analysis - Implementation Guide

## 🚢 Overview
This project compares **Classical (Node2Vec)** vs **Quantum Walk** embeddings for maritime logistics, port network analysis, anomaly detection, and shipment time prediction.

---

## 📊 How to Analyze the Data

### **Step 1: Run the Comprehensive Comparison**
```powershell
python compare_graph_methods.py
```

**What it does:**
- Loads the 100-port maritime network from `Datasets/adjacency_matrix.txt`
- Trains embeddings from classical and quantum random walks
- Compares embedding quality (clustering, separability)
- Detects anomalous ports using both methods
- Ranks ports by importance
- Analyzes route similarities

**Outputs:**
- `maritime_analysis/1_embedding_quality_comparison.png` - Quality metrics
- `maritime_analysis/2_embedding_space_visualization.png` - 2D PCA projections
- `maritime_analysis/3_anomaly_network_visualization.png` - Anomaly detection maps
- `maritime_analysis/4_port_importance_rankings.png` - Top ports by different metrics
- `maritime_analysis/comprehensive_port_analysis.csv` - Full port data
- `maritime_analysis/route_similarity_analysis.csv` - Route comparisons

---

## 🔍 Understanding the Comparisons

### **1. Embedding Quality (Part 1)**
Compares how well each method clusters ports by category:

**Metrics:**
- **Silhouette Score** (higher = better) - Measures cluster separation
- **Calinski-Harabasz Index** (higher = better) - Cluster definition quality
- **K-Means Inertia** (lower = better) - Within-cluster compactness

**Interpretation:**
- Higher scores = Better at distinguishing port types
- Better embeddings = More accurate predictions

---

### **2. Embedding Visualization (Part 2)**
PCA projections show how embeddings organize ports:

**What to look for:**
- **Clear clusters** = Good category separation
- **Distinct groups** = Method captures port hierarchy well
- **Quantum vs Classical differences** = Different structural patterns captured

**Real-world meaning:**
- Clustered ports = Similar shipping behavior
- Outliers = Unique connectivity patterns

---

### **3. Anomaly Detection (Part 3)**
Identifies unusual ports using Isolation Forest:

**Comparison:**
- **Both methods agree** = High-confidence anomalies (investigate first)
- **Node2Vec only** = Classical pattern violations (unusual local connections)
- **Quantum only** = Non-local pattern violations (unusual global position)
- **Agreement rate** = Overall method consistency

**Use cases:**
- Ports with unusual connectivity
- Potential bottlenecks
- Security risks (smuggling routes)
- Capacity issues

---

### **4. Port Importance (Part 4)**
Ranks ports by different importance measures:

**Metrics compared:**
- **Betweenness centrality** = Bridge between regions
- **PageRank** = Overall network influence
- **Node2Vec importance** = Classical walk frequency
- **Quantum importance** = Quantum walk probability

**Interpretation:**
- **High betweenness** = Critical transit points (failure causes delays)
- **High PageRank** = Major hubs (many connections)
- **Different rankings** = Methods capture different strategic importance

**Business value:**
- Resource allocation
- Capacity planning
- Risk mitigation
- Infrastructure investment

---

### **5. Route Similarity (Part 5)**
Analyzes shipping route patterns:

**Similarity Score:**
- High similarity = Ports have similar shipping patterns
- Can predict shipment times based on similar routes
- Helps optimize route selection

**Applications:**
- Route recommendation
- Delay prediction
- Alternative route finding

---

## 🎯 Key Findings Interpretation

### **When Node2Vec is Better:**
- Local network structure matters more
- Traditional shipping patterns dominate
- Community detection is priority

### **When Quantum Walks are Better:**
- Long-range dependencies exist
- Non-local effects matter (cascading delays)
- Exploring alternative patterns

### **When to Use Both:**
- Maximum confidence anomaly detection
- Comprehensive risk assessment
- Multi-perspective port ranking

---

## 💼 Business Applications

### **1. Anomaly Detection System**
```python
# High-priority alerts: Both methods agree
critical_ports = df[(df['n2v_anomaly']==1) & (df['q2_anomaly']==1)]

# Monitor: Single method flags
watch_ports = df[(df['n2v_anomaly']==1) | (df['q2_anomaly']==1)]
```

### **2. Shipment Time Prediction**
```python
# Use embedding similarity to predict delays
route_similarity = calculate_route_similarity(embedding, port_A, port_B)
predicted_delay_factor = 1 / (1 + route_similarity)
estimated_time = base_time * predicted_delay_factor
```

### **3. Port Resource Allocation**
```python
# Prioritize investment based on importance
critical_hubs = df.nlargest(10, 'n2v_importance')
backup_critical = df.nlargest(10, 'q2_importance')
investment_targets = set(critical_hubs) | set(backup_critical)
```

### **4. Risk Scoring System**
```python
# Multi-factor risk score
df['risk_score'] = (
    df['n2v_anomaly'] * 0.3 +
    df['q2_anomaly'] * 0.3 +
    (df['betweenness'] > threshold) * 0.2 +
    (df['degree'] < min_connections) * 0.2
)
```

---

## 📈 Advanced Analysis Options

### **Option A: Run with Different Parameters**
Modify `main.py` and `clustered_quantum.py` parameters:
- Increase `num_walks` for more data
- Adjust `walk_length` for longer patterns
- Change `p` and `q` in Node2Vec for different exploration

### **Option B: Real Maritime Data**
Replace `adjacency_matrix.txt` with actual:
- AIS vessel tracking data
- Port-to-port shipping records
- Supply chain network data

### **Option C: Time-Series Analysis**
Add temporal dimension:
- Monthly snapshots of network
- Track anomaly evolution
- Predict seasonal patterns

---

## 🛠️ Quick Start Commands

```powershell
# 1. Generate walks (if not already done)
python main.py                    # Classical walks
python clustered_quantum.py       # Quantum walks

# 2. Run comprehensive comparison
python compare_graph_methods.py

# 3. View results
cd maritime_analysis
ls *.png                          # List all visualizations
```

---

## 📋 Output Files Reference

| File | Description | Use Case |
|------|-------------|----------|
| `comprehensive_port_analysis.csv` | All port metrics + anomaly flags | Import to Excel/BI tools |
| `route_similarity_analysis.csv` | Route comparisons | Route optimization |
| `embedding_quality_metrics.csv` | Model performance | Method selection |
| `*.png` files | Visualizations | Reports & presentations |

---

## 🎓 Understanding Graph Metrics

### **Degree Centrality**
- Number of direct connections
- High degree = Major hub

### **Betweenness Centrality**
- How often port is on shortest paths
- High betweenness = Critical bridge

### **PageRank**
- Google's algorithm for web pages
- Measures overall importance in network

### **Embedding-based Importance**
- Learned from random walk patterns
- Captures structural role in network

---

## 🔬 Research Questions Answered

1. **Do quantum walks find different anomalies than classical?**
   - Yes - Check agreement rate in output
   - Quantum captures non-local patterns

2. **Which method better predicts port categories?**
   - Compare silhouette scores
   - Higher score = better category prediction

3. **Are rankings consistent across methods?**
   - Check correlation analysis in console output
   - High correlation = methods agree on importance

4. **Can embeddings predict shipment times?**
   - Yes - Use similarity scores
   - Similar embeddings = similar transit patterns

---

## 📞 Next Steps

1. **Review visualizations** in `maritime_analysis/` folder
2. **Analyze CSV outputs** for specific ports of interest
3. **Identify consensus anomalies** (both methods agree)
4. **Compare importance rankings** for strategic planning
5. **Build prediction models** using embeddings as features

---

## 🚨 Important Notes

- **Contamination rate** (15%) controls anomaly detection sensitivity
- **Alpha values** (0.1, 0.2, 0.3) control quantum behavior
- **Random seed (42)** ensures reproducible results
- **PCA** is for visualization only - use full embeddings for ML models
