# Technical Glossary - Maritime Network Analysis

## 📚 Complete Parameter & Term Reference

---

## **🔬 QUANTUM WALK PARAMETERS**

### **Alpha (α)**
- **Definition**: Controls the balance between quantum coherence and classical random walk behavior
- **Range**: 0.0 to 1.0
- **Used in**: `clustered_quantum.py`

**Mathematical Role**:
```python
H = (1 - α) * hamiltonian  # Coherent evolution term
c_ops = [√α * L]           # Classical jump operators
```

**Interpretation**:
- **α = 0.0**: Pure quantum (coherent evolution only, no jumps)
- **α = 0.1**: 90% quantum, 10% classical (minimal decoherence)
- **α = 0.5**: Balanced quantum-classical hybrid
- **α = 1.0**: Pure classical random walk (maximum decoherence)

**Current Values**: `[0.1, 0.2, 0.3]` in code
**Recommended**: `[0.1, 0.3, 0.5, 0.7]` for better range

**Business Impact**: Higher α captures local patterns; lower α captures non-local quantum interference effects

---

### **Evolution Time (total_time)**
- **Definition**: Duration of quantum walk simulation in arbitrary time units
- **Current Value**: `14.5` seconds
- **Used in**: `clustered_quantum.py`

**Purpose**: Controls how many quantum jumps occur during the walk

**Formula**:
```
Average Jumps ≈ Evolution Time × Average Transition Rate
```

**Current Issue**: Only produces ~3-4 jumps per walk
**Recommended**: `50.0` seconds for ~10 jumps (matching classical walks)

**Why It Matters**: Longer time → more exploration → better embeddings

---

### **Number of Trajectories (num_traj)**
- **Definition**: How many independent quantum walks start from each node
- **Current Value**: `10`
- **Used in**: `clustered_quantum.py`, `config.txt`

**Purpose**: Statistical sampling of quantum walk outcomes

**Similar to**: Classical `num_walks` parameter
**Trade-off**: More trajectories = better statistics but slower computation

---

### **Hamiltonian (H)**
- **Definition**: Quantum operator governing coherent evolution of the system
- **Matrix Form**: Same as adjacency matrix in this implementation
- **Used in**: `prepare_quantum_operators()`

**Physics**:
```
H|ψ⟩ = A|ψ⟩  where A = adjacency matrix
```

**Interpretation**: Determines quantum amplitude flow through network edges

---

### **Collapse Operators (c_ops)**
- **Definition**: Quantum jump operators that cause transitions between nodes
- **Physical Meaning**: Measurement/decoherence events that "collapse" quantum state
- **Used in**: `prepare_quantum_operators()`, `mcsolve()`

**Construction**:
```python
L_ij = √(transition_probability) * |j⟩⟨i|
```

**Role**: Enable classical-like jumps in the quantum walk, controlled by α

---

### **Monte Carlo Wave Function (mcsolve)**
- **Definition**: Numerical method for simulating open quantum systems
- **Library**: QuTip (Quantum Toolbox in Python)
- **Used in**: `quantum_jumps_single_node()`

**What It Does**: Simulates random quantum trajectories with stochastic jumps

**Alternative Methods**: 
- Master equation (slower but deterministic)
- Quantum trajectory method (equivalent)

---

## **🎯 CLASSICAL NODE2VEC PARAMETERS**

### **p (Return Parameter)**
- **Definition**: Controls likelihood of immediately revisiting the previous node
- **Current Value**: `3.0`
- **Used in**: `main.py`, `node2vec.py`

**Interpretation**:
- **p > 1**: Discourage backtracking (less likely to return)
- **p = 1**: No bias
- **p < 1**: Encourage backtracking

**Effect**:
- High p (like 3.0) → Explores outward, finds structural equivalence
- Low p → Stays local, finds tight communities

**Current Setting Analysis**: `p=3.0` biases toward exploring new regions (good for global structure)

---

### **q (In-Out Parameter)**
- **Definition**: Controls exploration strategy (BFS vs DFS)
- **Current Value**: `0.5`
- **Used in**: `main.py`, `node2vec.py`

**Interpretation**:
- **q > 1**: Favor inward exploration (DFS-like, explores neighborhoods deeply)
- **q = 1**: No bias
- **q < 1**: Favor outward exploration (BFS-like, explores broadly)

**Effect**:
- High q → Micro-level community detection
- Low q (like 0.5) → Macro-level homophily detection

**Current Setting Analysis**: `q=0.5` explores broadly (good for hub detection)

---

### **num_walks**
- **Definition**: Number of random walks starting from each node
- **Current Value**: `2` (from `config.txt`)
- **Used in**: `main.py`, `node2vec.py`

**Purpose**: Statistical sampling to capture diverse paths

**Total Walks**: `num_walks × num_nodes = 2 × 100 = 200` walks

**Trade-off**: 
- More walks → Better embeddings, longer training
- Fewer walks → Faster but might miss patterns

**Recommended**: 5-10 for production systems

---

### **walk_length (desired_jumps)**
- **Definition**: Number of steps in each random walk
- **Current Value**: `10` (from `config.txt`)
- **Used in**: `main.py`, `node2vec.py`

**Purpose**: Determines context window for each node

**Interpretation**: 
- Longer walks → Capture global structure
- Shorter walks → Focus on local neighborhoods

**Current Setting**: `10` is standard for medium-sized networks

---

### **Alias Sampling**
- **Definition**: Efficient algorithm for sampling from discrete probability distributions
- **Time Complexity**: O(1) per sample after O(n) preprocessing
- **Used in**: `node2vec.py` (`alias_setup()`, `alias_draw()`)

**Purpose**: Fast node selection during biased random walks

**Alternative**: Direct sampling O(n) per sample (much slower)

---

## **🧠 EMBEDDING PARAMETERS**

### **vector_size (embedding_dimension)**
- **Definition**: Dimensionality of learned node embeddings
- **Current Value**: `64` (from `config.txt`)
- **Used in**: Word2Vec training in all scripts

**Purpose**: Size of vector representation for each node/port

**Trade-offs**:
- Higher dimensions → More expressive but overfitting risk
- Lower dimensions → Faster but information loss

**Optimal Range**: For 100 nodes, 64-128 dimensions

**Formula**: `8 × log₂(num_nodes)` to `10 × √(num_nodes)`

---

### **window (window_size)**
- **Definition**: Context window size for Word2Vec training
- **Current Value**: `5` (from `config.txt`)
- **Used in**: Word2Vec model in all scripts

**Purpose**: How many neighbor nodes in walk are considered "context"

**In Skip-Gram**: Predicts context nodes given target node

**Example Walk**: `[1, 5, 3, 9, 2]` with window=2
- Node 3 context: `{5, 1, 9, 2}` (2 before, 2 after)

**Interpretation**:
- Larger window → Captures broader patterns
- Smaller window → Focuses on immediate neighbors

---

### **Skip-Gram (sg=1)**
- **Definition**: Word2Vec training algorithm
- **Value**: `sg=1` (in `compare_graph_methods.py`)
- **Alternative**: CBOW (sg=0)

**How It Works**:
- Input: Target node
- Output: Predict context nodes
- Loss: Maximize probability of context given target

**Why Skip-Gram for Graphs**: Better for rare nodes (less frequent paths)

---

### **epochs (iter)**
- **Definition**: Number of training iterations over the walk dataset
- **Current Value**: `10` in comparison script, `1` in main.py
- **Used in**: Word2Vec training

**Purpose**: How many times to process all walks

**Trade-off**: More epochs → Better convergence but risk overfitting

---

## **📊 EVALUATION METRICS**

### **Silhouette Score**
- **Definition**: Measures how well-separated clusters are
- **Range**: -1 (worst) to +1 (best)
- **Formula**: `s = (b - a) / max(a, b)`
  - a = avg distance to same cluster
  - b = avg distance to nearest other cluster

**Interpretation**:
- s > 0.5: Strong clustering
- s > 0.25: Reasonable clustering
- s < 0: Poor clustering (overlap)

**Used For**: Comparing embedding quality

---

### **Calinski-Harabasz Index**
- **Definition**: Ratio of between-cluster to within-cluster variance
- **Range**: 0 to ∞ (higher is better)
- **Formula**: `CH = (SSB/SSW) × ((n-k)/(k-1))`
  - SSB = between-cluster sum of squares
  - SSW = within-cluster sum of squares

**Interpretation**: Higher values = denser, better separated clusters

---

### **K-Means Inertia**
- **Definition**: Sum of squared distances to nearest cluster center
- **Range**: 0 to ∞ (lower is better)
- **Used For**: Measuring cluster compactness

**Formula**: `Inertia = Σ min ||xᵢ - μⱼ||²`

**Why Lower is Better**: Points closer to cluster centers

---

### **Contamination Rate**
- **Definition**: Expected proportion of anomalies in dataset
- **Current Value**: `0.15` (15%)
- **Used in**: Isolation Forest anomaly detection

**Purpose**: Sensitivity threshold for anomaly detection

**Effect**:
- High contamination (0.2) → More ports flagged as anomalous
- Low contamination (0.05) → Only extreme outliers flagged

**Tuning**: Should match real-world anomaly prevalence

---

### **Isolation Forest**
- **Definition**: Anomaly detection algorithm based on random trees
- **Principle**: Anomalies are easier to isolate (fewer splits needed)
- **Used in**: `compare_graph_methods.py`

**How It Works**:
1. Build random trees with random splits
2. Measure path length to isolate each point
3. Shorter paths → Anomalies

**Advantages**: Fast, works well in high dimensions

---

## **🌐 GRAPH THEORY METRICS**

### **Degree Centrality**
- **Definition**: Number of edges connected to a node
- **Formula**: `degree(v) = |{u : (v,u) ∈ E}|`
- **Used For**: Identifying hub ports

**Interpretation**: High degree = many direct connections

---

### **Betweenness Centrality**
- **Definition**: Fraction of shortest paths passing through a node
- **Formula**: `CB(v) = Σ (σst(v)/σst)`
  - σst = total shortest paths from s to t
  - σst(v) = paths through v

**Used For**: Finding critical transit points

**Business Impact**: High betweenness nodes are bottlenecks (failure causes disruption)

---

### **Closeness Centrality**
- **Definition**: Inverse of average shortest path to all other nodes
- **Formula**: `CC(v) = (n-1) / Σ d(v,u)`

**Interpretation**: How quickly a node can reach all others

**Use Case**: Optimal port locations for distribution

---

### **PageRank**
- **Definition**: Google's algorithm for measuring node importance
- **Principle**: Important nodes are connected to other important nodes
- **Formula**: `PR(v) = (1-d)/n + d × Σ PR(u)/degree(u)`

**Damping Factor (d)**: Typically 0.85

**Use Case**: Overall strategic importance in network

---

### **Eigenvector Centrality**
- **Definition**: Importance based on importance of neighbors
- **Formula**: Principal eigenvector of adjacency matrix
- **Relation**: PageRank is a variant of this

**Interpretation**: Connected to well-connected nodes

---

## **🔧 TECHNICAL IMPLEMENTATION TERMS**

### **PCA (Principal Component Analysis)**
- **Definition**: Dimensionality reduction technique
- **Used in**: Visualizing 64D embeddings in 2D plots
- **Purpose**: Project high-dimensional embeddings to 2D for plotting

**Important**: PCA is for visualization only; use full embeddings for ML models

---

### **StandardScaler**
- **Definition**: Normalizes features to zero mean, unit variance
- **Formula**: `z = (x - μ) / σ`
- **Used in**: Before K-Means clustering

**Why Needed**: Different features have different scales

---

### **Cosine Similarity**
- **Definition**: Measures angle between two vectors
- **Formula**: `sim = (A·B) / (||A|| ||B||)`
- **Range**: -1 to +1 (1 = identical direction)

**Used For**: Route similarity in embedding space

---

### **L2 Norm (Euclidean Norm)**
- **Definition**: Vector magnitude/length
- **Formula**: `||v|| = √(v₁² + v₂² + ... + vₙ²)`
- **Used For**: Calculating embedding importance scores

---

### **Pearson Correlation**
- **Definition**: Linear correlation coefficient
- **Range**: -1 to +1
- **Used For**: Comparing ranking methods

**Interpretation**:
- r = 1: Perfect positive correlation
- r = 0: No correlation
- r = -1: Perfect negative correlation

---

### **Spearman Correlation**
- **Definition**: Rank-order correlation
- **Difference from Pearson**: Uses ranks instead of raw values
- **Robust to**: Outliers and non-linear relationships

**When to Use**: Comparing ordinal rankings (1st, 2nd, 3rd...)

---

## **📁 FILE FORMAT SPECIFICATIONS**

### **Walk Files (.txt)**
- **Format**: One walk per line, space-separated node IDs
- **Example**:
  ```
  0 5 12 8 15
  3 7 9 2 1
  ```
- **Files**: `2ndRW_walks_*.txt`, `Qwalks_alpha_*.txt`

### **Adjacency Matrix (adjacency_matrix.txt)**
- **Format**: Space-separated matrix, 1 = edge exists, 0 = no edge
- **Size**: N×N where N = number of nodes
- **Properties**: Symmetric (undirected graph)

### **Node Colors (node_colors.txt)**
- **Format**: `node_id category_id`
- **Example**:
  ```
  0 2
  1 0
  2 1
  ```
- **Purpose**: Ground truth port categories for evaluation

---

## **⚙️ CONFIGURATION FILE (config.txt)**

```plaintext
num_traj=2              # Number of walks per node (classical)
desired_jumps=10        # Walk length
window=5                # Word2Vec context window
embedding_dimension=64  # Embedding vector size
```

**Relationship**:
- `num_traj` → `num_walks` in Node2Vec
- `desired_jumps` → `walk_length` in Node2Vec

---

## **🎯 QUICK REFERENCE TABLE**

| Parameter | Classical Value | Quantum Value | Impact on Results |
|-----------|----------------|---------------|-------------------|
| **α** | N/A | 0.1, 0.2, 0.3 | Quantum vs classical behavior |
| **p** | 3.0 | N/A | Exploration vs backtracking |
| **q** | 0.5 | N/A | BFS vs DFS search |
| **num_walks/traj** | 2 | 10 | Statistical quality |
| **walk_length** | 10 | ~3-4 (adaptive) | Context size |
| **embedding_dim** | 64 | 64 | Expressiveness |
| **window** | 5 | 5 | Local context |
| **contamination** | 0.15 | 0.15 | Anomaly sensitivity |

---

## **🔍 UNDERSTANDING THE PHYSICS**

### **Quantum Coherence**
- **Definition**: Ability of quantum system to maintain superposition
- **In Walks**: Node can be in multiple locations simultaneously
- **Controlled by**: α parameter (lower = more coherence)

### **Decoherence**
- **Definition**: Loss of quantum properties due to environment interaction
- **In Walks**: Collapse to classical position (quantum jump)
- **Controlled by**: α parameter (higher = more decoherence)

### **Quantum Interference**
- **Definition**: Amplitudes can add constructively or destructively
- **Effect**: Some paths become more/less probable than classical
- **Result**: Quantum walks find different patterns than classical

---

## **📖 NOTATION GUIDE**

- `|ψ⟩`: Quantum state vector (Dirac notation)
- `⟨i|`: Bra vector (row vector)
- `|j⟩`: Ket vector (column vector)
- `⟨i|j⟩`: Inner product
- `|j⟩⟨i|`: Outer product (matrix)
- `Â`: Operator (hat notation)
- `H`: Hamiltonian operator
- `ρ`: Density matrix
- `σ`: Standard deviation or shortest path count
- `α`: Alpha parameter
- `μ`: Mean
- `ε`: Epsilon (small number)

---

## **💡 PRACTICAL TIPS**

### **When to Increase α:**
- Need more classical-like behavior
- Graph has clear community structure
- Want local pattern detection

### **When to Decrease α:**
- Want to find non-local correlations
- Detecting long-range dependencies
- Exploring quantum advantage

### **When to Increase p:**
- Want structural equivalence (similar roles)
- Avoid redundant backtracking
- Global pattern focus

### **When to Increase q:**
- Want homophily (similar attributes cluster)
- Need deep community exploration
- Local pattern focus

---

## **🚨 COMMON PITFALLS**

1. **α = 1.0**: Pure classical random walk (no quantum advantage)
2. **Low evolution_time**: Quantum walks too short (poor coverage)
3. **High contamination**: Too many false positive anomalies
4. **Small embedding_dim**: Information loss
5. **Low num_walks**: Poor statistical sampling
6. **Mismatched walk lengths**: Unfair comparison between methods

---

## **📚 REFERENCES**

- **Node2Vec Paper**: Grover & Leskovec, KDD 2016
- **Quantum Walks**: Kempe, Contemporary Physics 2003
- **QuTip Documentation**: qutip.org
- **Isolation Forest**: Liu et al., ICDM 2008
- **Word2Vec**: Mikolov et al., NIPS 2013

---

*Last Updated: November 22, 2025*
*For implementation details, see code files and `MARITIME_ANALYSIS_GUIDE.md`*
