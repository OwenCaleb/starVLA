
# Experiment Plan for Mechanism Analysis

## 0. Scope

Implement only the following two tests:

1. **Stage-wise token UMAP + unsupervised separation metrics**
2. **Stage-wise final-layer group-level Sankey / alluvial flow based on the attention matrix**

Do **not** implement:

* attention rollout
* attention-flow DAG
* teacher-correspondence heatmap
* reconstruction-based analysis
* probe training

---

## 1. Common Setup

### 1.1 Checkpoints

Use the **last checkpoint** from each training stage:

* Stage I checkpoint
* Stage II checkpoint
* Stage III checkpoint

### 1.2 Input data

Use one prepared episode folder containing sampled images.
The script should:

* read all images in temporal order
* run inference on each image independently
* collect hidden states / attention statistics for each image
* average statistics across images when required

### 1.3 Token groups

Use the following token groups throughout:

* `vis`
* `lang`
* `sub`
* `spa`
* `dyn`
* `act`

---

## 2. Test 1: Stage-wise Token UMAP

### 2.1 Goal

Visualize how token representations evolve across stages.

Main version:

* **all tokens** are treated as points

Backup version:

* **slot-only tokens** (`sub`, `spa`, `dyn`, `act`) are treated as points

### 2.2 Features to extract

For each stage checkpoint and each image:

* run forward pass
* extract the **last-layer hidden states**
* treat **each token** as one point
* save metadata:

  * image id
  * stage
  * token index
  * token group
  * hidden vector

### 2.3 UMAP

Use one global UMAP reducer fit on the concatenation of Stage I/II/III vectors, then transform each stage separately.

### 2.4 Unsupervised metrics

Compute in the **original hidden space**:

* **Silhouette score**
* **Inter/Intra distance ratio**

Inter/Intra ratio definition:

* intra = average pairwise distance among tokens with the same token-group label
* inter = average pairwise distance among tokens with different token-group labels
* ratio = inter / intra

Compute metrics separately for:

* Stage I
* Stage II
* Stage III

### 2.5 Required outputs

Export **three separate UMAP files**:

* `umap_stage1.csv`
* `umap_stage2.csv`
* `umap_stage3.csv`

Required columns:

* `image_id`
* `token_id`
* `token_group`
* `umap_x`
* `umap_y`

Example:

```csv
image_id,token_id,token_group,umap_x,umap_y
img_0001,123,vis,-1.24,2.31
img_0001,124,vis,-1.11,2.08
img_0001,315,sub,0.42,-1.73
img_0005,317,act,1.82,0.64
```

Export **three separate metric files**:

* `token_separation_metrics_stage1.csv`
* `token_separation_metrics_stage2.csv`
* `token_separation_metrics_stage3.csv`

Required columns:

* `metric`
* `value`

Example:

```csv
metric,value
silhouette,0.26
inter_intra_ratio,1.87
```

### 2.6 ChiPlot templates

Use:

* **Scatter plot** for each UMAP file
* **Bar plot** for each stage metric file, or combine the three metric files later into one bar-plot table if needed for plotting

---

## 3. Test 2: Stage-wise Final-Layer Q→K Sankey / Alluvial Flow

### 3.1 Goal

Visualize how different **query token groups** attend to different **key token groups** in the **final layer**.

Convention:

* **Left column = Q groups**
* **Right column = K groups**

### 3.2 Attention to use

Use the **final attention weights after softmax**, not raw `QK^T` scores.

If multi-head attention is returned:

* first average over all heads
* then aggregate at the token-group level

### 3.3 Group-level aggregation

Let the final-layer, head-averaged attention matrix be `A`, where rows are **query tokens** and columns are **key tokens**.

For each pair `(q_group, k_group)`, compute the average attention mass from tokens in `q_group` to tokens in `k_group`.

Do this for each image, then average over all images in the episode folder.

### 3.4 Required outputs

Export **three separate Sankey files**:

* `sankey_qk_stage1.csv`
* `sankey_qk_stage2.csv`
* `sankey_qk_stage3.csv`

Required columns:

* `source`
* `target`
* `weight`

where:

* `source = q_group`
* `target = k_group`

Example:

```csv
source,target,weight
vis,vis,0.31
vis,lang,0.08
vis,sub,0.04
lang,lang,0.42
sub,spa,0.17
act,sub,0.28
act,spa,0.21
act,dyn,0.19
act,vis,0.14
act,lang,0.09
```

Optional long-form export:

* `qk_flow_stage1_long.csv`
* `qk_flow_stage2_long.csv`
* `qk_flow_stage3_long.csv`

Required columns:

* `image_id`
* `q_group`
* `k_group`
* `weight`

### 3.5 ChiPlot templates

Use:

* **Sankey plot** for the main visualization
* **Heatmap** as optional backup if Sankey becomes too dense

If exporting heatmap-ready files, use:

* `qk_heatmap_stage1.csv`
* `qk_heatmap_stage2.csv`
* `qk_heatmap_stage3.csv`

Required columns:

* `q_group`
* `k_group`
* `weight`

---

## 4. Final Deliverables

### Test 1

* `umap_stage1.csv`
* `umap_stage2.csv`
* `umap_stage3.csv`
* `token_separation_metrics_stage1.csv`
* `token_separation_metrics_stage2.csv`
* `token_separation_metrics_stage3.csv`

### Test 2

* `sankey_qk_stage1.csv`
* `sankey_qk_stage2.csv`
* `sankey_qk_stage3.csv`

### Optional backup files

* `qk_flow_stage1_long.csv`
* `qk_flow_stage2_long.csv`
* `qk_flow_stage3_long.csv`
* `qk_heatmap_stage1.csv`
* `qk_heatmap_stage2.csv`
* `qk_heatmap_stage3.csv`

---

## 5. Implementation Notes

1. Use the same episode image folder for all three checkpoints.
2. Keep token-group assignment deterministic and based on the actual model layout.
3. Save intermediate outputs before plotting.
4. Separate code into:

   * feature extraction
   * aggregation
   * CSV export
   * optional plotting
5. Make it easy to rerun on another checkpoint trio or another image folder.
