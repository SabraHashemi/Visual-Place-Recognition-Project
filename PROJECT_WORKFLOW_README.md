# Visual Place Recognition Project - Workflow Guide

## Overview

This project implements a **multi-phase Visual Place Recognition (VPR) pipeline** for identifying locations from images. The workflow consists of four distinct phases that can be run sequentially to progressively improve place recognition accuracy through initial retrieval, image matching, re-ranking, and uncertainty evaluation.

## Project Structure

The project follows a **modular, phase-based architecture** where each phase is executed independently. This design provides flexibility, allows for easy debugging, and enables users to experiment with different methods at each stage.

## Why No Automated Pipeline?

The project intentionally **does not include a single automated pipeline script** for the following reasons:

1. **Flexibility**: Each phase can be run independently, allowing users to:
   - Skip phases if not needed
   - Experiment with different methods at each stage
   - Re-run specific phases with different parameters without re-running everything

2. **Debugging & Development**: Modular execution makes it easier to:
   - Debug individual phases
   - Inspect intermediate results
   - Modify parameters between phases

3. **Resource Management**: Users can:
   - Run phases on different machines/resources
   - Pause between phases to analyze results
   - Reuse results from previous phases

4. **Research & Experimentation**: The modular design supports:
   - Comparing different method combinations
   - A/B testing different configurations
   - Iterative refinement of each phase

If you need automation, you can easily create a wrapper script that calls all phases sequentially.

---

## The Four Phases

### Phase 1: VPR Evaluation (Initial Retrieval)

**Script**: `VPR-methods-evaluation/main.py`

**Purpose**: Performs initial place recognition by extracting global image descriptors and finding the most similar database images for each query.

**What it does**:
1. Loads a VPR model (e.g., CosPlace, NetVLAD, etc.)
2. Extracts descriptors from database images
3. Extracts descriptors from query images
4. Uses FAISS for efficient similarity search
5. Computes recall metrics (R@1, R@5, R@10, R@20)
6. Saves top-k predictions for each query

**Output**:
- Text files (`*.txt`) containing query image paths and top-k prediction paths
- Recall metrics (printed and logged to wandb if enabled)
- Optional: Descriptors, visualizations, and uncertainty data

**Where to Find Recall Values (Before Reranking)**:
The recall metrics from Phase 1 are displayed in two places:
1. **Console Output**: During execution, you'll see a line like:
   ```
   R@1: 65.2, R@5: 82.5, R@10: 88.3, R@20: 92.1
   ```
2. **Log File**: Saved in `logs/<log_dir>/<timestamp>/info.log`
   - Open this file and search for "R@" to find the recall metrics
   - Example: `2024-01-15 10:30:45 R@1: 65.2, R@5: 82.5, R@10: 88.3, R@20: 92.1`

**Important**: Recall values are only computed if you have labels (don't use `--no_labels` flag). If you used `--no_labels`, Phase 1 won't compute or display recall metrics.

**Key Parameters**:
- `--method`: VPR method to use
- `--backbone`: Neural network backbone
- `--descriptors_dimension`: Dimension of feature descriptors
- `--num_preds_to_save`: Number of top predictions to save
- `--recall_values`: Recall metrics to compute (e.g., 1, 5, 10, 20)

**Example Command**:
```bash
python VPR-methods-evaluation/main.py \
--num_workers 8 \
--batch_size 32 \
--log_dir log_dir \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
--image_size 512 512 \
--database_folder '<path-to-database-folder>' \
--queries_folder '<path-to-queries-folder>' \
--num_preds_to_save 20 \
--recall_values 1 5 10 20 \
--save_for_uncertainty
```

---

### Phase 2: Image Matching on Retrieval Results

**Script**: `match_queries_preds.py`

**Purpose**: Performs detailed image matching between query images and their top-k predictions from Phase 1. This identifies keypoint correspondences and geometric matches.

**What it does**:
1. Loads an image matcher (e.g., SuperPoint-LightGlue, SIFT-LightGlue, etc.)
2. For each query and its top-k predictions:
   - Extracts keypoints and descriptors
   - Finds correspondences between query and prediction
   - Computes number of inliers (geometrically consistent matches)
3. Saves matching results for re-ranking

**Output**:
- Torch files (`*.torch`) containing matching results for each query
- Each file contains keypoint matches and inlier counts for all top-k predictions

**Key Parameters**:
- `--preds-dir`: Directory containing Phase 1 predictions (`.txt` files)
- `--matcher`: Image matching method to use
- `--num-preds`: Number of predictions to match (should match Phase 1's `num_preds_to_save`)
- `--device`: Computing device (cpu/cuda)

**Example Command**:
```bash
python match_queries_preds.py \
--preds-dir '<path-to-predictions-folder>' \
--matcher 'superpoint-lg' \
--device 'cuda' \
--num-preds 20
```

---

### Phase 3: Re-ranking Performance Evaluation

**Script**: `reranking.py`

**Purpose**: Re-ranks the initial predictions from Phase 1 based on the number of inliers from Phase 2, then evaluates the improved recall metrics.

**What it does**:
1. Loads predictions from Phase 1
2. Loads matching results (inlier counts) from Phase 2
3. For each query:
   - Sorts predictions by number of inliers (descending)
   - Re-evaluates recall metrics with re-ranked predictions
4. Computes and displays improved recall@k metrics

**Output**:
- Recall metrics after re-ranking (printed to console)
- Shows improvement over Phase 1 baseline

**Comparing Before vs After Reranking**:
To see the improvement, you need to compare:
- **Before (Phase 1)**: Check the console output or `logs/<log_dir>/<timestamp>/info.log` file
- **After (Phase 3)**: Check the console output from running `reranking.py`

Example comparison:
```
Phase 1 (Before): R@1: 65.2, R@5: 82.5, R@10: 88.3, R@20: 92.1
Phase 3 (After):  R@1: 78.5, R@5: 91.2, R@10: 94.8, R@20: 96.5
```

**Key Parameters**:
- `--preds-dir`: Directory with Phase 1 predictions
- `--inliers-dir`: Directory with Phase 2 matching results
- `--num-preds`: Number of predictions to re-rank
- `--recall-values`: Recall metrics to compute
- `--positive-dist-threshold`: Distance threshold (meters) for positive matches

**Example Command**:
```bash
python reranking.py \
--preds-dir '<path-to-predictions-folder>' \
--inliers-dir '<path-to-inliers-folder>' \
--num-preds 20 \
--recall-values 1 5 10 20
```

---

### Phase 4: Uncertainty Evaluation [Optional - AML Students]

**Script**: `python -m vpr_uncertainty.eval`

**Purpose**: Evaluates uncertainty estimation methods for VPR predictions, computing metrics like AUCPR (Area Under Precision-Recall curve).

**What it does**:
1. Loads predictions, distances, and matching results
2. Computes uncertainty scores using various baselines:
   - L2 distance
   - Prediction Agreement (PA)
   - Spatial Uncertainty Estimation (SUE)
   - Random baseline
3. Evaluates uncertainty quality using precision-recall curves

**Output**:
- AUCPR metrics for different uncertainty estimation methods
- Comparison of uncertainty estimation approaches

**Key Parameters**:
- `--preds-dir`: Directory with Phase 1 predictions
- `--inliers-dir`: Directory with Phase 2 matching results
- `--z-data-path`: Path to `z_data.torch` file from Phase 1 (requires `--save_for_uncertainty` flag)

**Example Command**:
```bash
python -m vpr_uncertainty.eval \
--preds-dir '<path-to-predictions-folder>' \
--inliers-dir '<path-to-inliers-folder>' \
--z-data-path '<path-to-z-data-file>'
```

---

## Available Methods

### VPR Methods (Phase 1)

The following VPR methods are available via the `--method` argument:

**Classical Methods**:
- `netvlad` - NetVLAD descriptor aggregation
- `apgem` - APGeM method
- `sfrs` - SFRS method

**Modern Deep Learning Methods**:
- `cosplace` - CosPlace (supports VGG16, ResNet18/50/101/152)
- `convap` - Conv-AP (ResNet50)
- `mixvpr` - MixVPR (ResNet50)
- `eigenplaces` - EigenPlaces (multiple backbones)
- `eigenplaces-indoor` - Indoor variant of EigenPlaces

**DINOv2-based Methods**:
- `anyloc-urban` - AnyLoc for urban scenes
- `anyloc-indoor` - AnyLoc for indoor scenes
- `anyloc-aerial` - AnyLoc for aerial images
- `anyloc-structured` - AnyLoc for structured environments
- `anyloc-unstructured` - AnyLoc for unstructured environments
- `anyloc-global` - AnyLoc global variant
- `salad` - SALAD method
- `salad-indoor` - SALAD indoor variant
- `cricavpr` - CRICA-VPR
- `clique-mining` - Clique Mining
- `megaloc` - MegaLoc
- `boq` - Bag of Queries (ResNet50 or DINOv2)
- `dinomix` - DINO-Mix

**Backbone Options**:
- `VGG16`
- `ResNet18`, `ResNet50`, `ResNet101`, `ResNet152`
- `Dinov2` (DINOv2)

### Image Matching Methods (Phase 2)

The following image matchers are available via the `--matcher` argument:

**Dense Matchers**:
- `roma`, `tiny-roma` - RoMa dense matching
- `minima-roma` - MINIMA with RoMa

**Semi-dense Matchers**:
- `loftr`, `eloftr`, `se2loftr`, `xoftr` - LoFTR variants
- `aspanformer`, `matchformer` - Transformer-based matchers
- `minima-loftr` - MINIMA with LoFTR

**Sparse Matchers** (LightGlue-based):
- `sift-lg` - SIFT with LightGlue
- `superpoint-lg` - SuperPoint with LightGlue
- `disk-lg` - DISK with LightGlue
- `aliked-lg` - ALIKED with LightGlue
- `doghardnet-lg` - DoG-HardNet with LightGlue
- `dedode-lg` - DeDoDe with LightGlue
- `xfeat-lg` - XFeat with LightGlue

**Other Sparse Matchers**:
- `dedode` - DeDoDe standalone
- `steerers`, `affine-steerers` - Steerers variants
- `xfeat`, `xfeat-star` - XFeat variants
- `omniglue` - OmniGlue
- `superglue` - SuperGlue
- `patch2pix` - Patch2Pix
- `r2d2`, `d2net` - Other feature detectors
- `duster` - DUSTER
- `master` - MASt3R
- `gim-dkm`, `gim-lg` - GIM variants
- `sift-sphereglue`, `superpoint-sphereglue` - SphereGlue variants
- `minima-splg` - MINIMA with SuperPoint-LightGlue

**Subpixel Matchers**:
- `xfeat-subpx`, `xfeat-lg-subpx`
- `dedode-subpx`
- `splg-subpx`, `aliked-subpx`

---

## Complete Workflow Example

Here's a typical workflow from start to finish:

### 1. Setup
```bash
# Install dependencies
cd image-matching-models
pip install -e .[all]
pip install faiss-cpu wandb

# Download datasets
cd ..
python download_datasets.py
```

### 2. Phase 1: VPR Evaluation
```bash
python VPR-methods-evaluation/main.py \
--method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
--database_folder 'datasets/pitts30k/database' \
--queries_folder 'datasets/pitts30k/queries' \
--num_preds_to_save 20 \
--recall_values 1 5 10 20 \
--save_for_uncertainty \
--log_dir my_experiment
```

**Output**: `logs/my_experiment/<timestamp>/` containing:
- `*.txt` files with predictions
- `z_data.torch` (if `--save_for_uncertainty` used)

### 3. Phase 2: Image Matching
```bash
python match_queries_preds.py \
--preds-dir 'logs/my_experiment/<timestamp>' \
--matcher 'superpoint-lg' \
--device 'cuda' \
--num-preds 20
```

**Output**: `logs/my_experiment/<timestamp>_superpoint-lg/` containing:
- `*.torch` files with matching results

### 4. Phase 3: Re-ranking
```bash
python reranking.py \
--preds-dir 'logs/my_experiment/<timestamp>' \
--inliers-dir 'logs/my_experiment/<timestamp>_superpoint-lg' \
--num-preds 20 \
--recall-values 1 5 10 20
```

**Output**: Console output showing improved recall metrics

### 5. Phase 4: Uncertainty Evaluation (Optional)
```bash
python -m vpr_uncertainty.eval \
--preds-dir 'logs/my_experiment/<timestamp>' \
--inliers-dir 'logs/my_experiment/<timestamp>_superpoint-lg' \
--z-data-path 'logs/my_experiment/<timestamp>/z_data.torch'
```

---

## Weights & Biases Integration

The project includes automatic logging to Weights & Biases (wandb) for Phase 1. Metrics and hyperparameters are automatically logged if:
1. `wandb` is installed
2. You're logged in (`wandb login`)

**Logged Information**:
- Hyperparameters (method, backbone, batch size, etc.)
- Recall metrics (R@1, R@5, R@10, R@20)
- Experiment name and configuration

To disable wandb logging, set environment variable: `WANDB_DISABLED=true`

---

## Dataset Format

Images should follow this naming convention:
```
@ UTM_easting @ UTM_northing @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension
```

**Required**: UTM coordinates (easting, northing, zone_number, zone_letter)
**Optional**: Other fields can be empty

Example: `@123456.78@4567890.12@33@U@40.7128@-74.0060@@@0@0@0@1.5@@@.jpg`

---

## Tips & Best Practices

1. **Start Small**: Test with a small subset of data before running full evaluation
2. **Method Selection**: 
   - For general use: `cosplace` or `eigenplaces`
   - For indoor: `eigenplaces-indoor` or `anyloc-indoor`
   - For aerial: `anyloc-aerial`
3. **Image Matching**: 
   - `superpoint-lg` is a good default for most cases
   - `loftr` or `roma` for challenging cases with large viewpoint changes
4. **Resource Management**: 
   - Phase 1 is the most computationally intensive
   - Phase 2 can be parallelized by processing queries in batches
5. **Experiment Tracking**: Use wandb to compare different method combinations

---

## Troubleshooting

**Issue**: Import errors for wandb
- **Solution**: Install with `pip install wandb` or set `WANDB_DISABLED=true`

**Issue**: CUDA out of memory
- **Solution**: Reduce `--batch_size` in Phase 1 or use CPU for Phase 2

**Issue**: Missing predictions between phases
- **Solution**: Ensure `--num-preds` in Phase 2 matches `--num_preds_to_save` from Phase 1

**Issue**: Path not found errors
- **Solution**: Use absolute paths or ensure you're in the correct directory

---

## Next Steps

1. **Experiment**: Try different VPR methods and image matchers
2. **Compare**: Use wandb to track and compare experiments
3. **Optimize**: Tune hyperparameters for your specific dataset
4. **Extend**: Add custom methods or evaluation metrics

For questions or issues, refer to the main [README.md](./README.md) or the individual module documentation.

---

## Understanding Reranking: A Detailed Explanation

If you're confused about what reranking does and why it improves results, this section provides a detailed explanation with examples.

### The Problem with Phase 1

**Phase 1** ranks predictions based on **global descriptor similarity** (using FAISS distance). While this is fast and works reasonably well, it's not perfect:

- Similar-looking places can rank higher than the actual match
- The correct match might be at position 5 or 10, not position 1
- Global descriptors capture overall appearance but miss fine geometric details

**Example Scenario**: 
```
Query: Image of Eiffel Tower

Phase 1 Top-5 (ranked by descriptor similarity):
  1. Some random tower (wrong, but similar appearance) ❌
  2. Another tower (wrong) ❌
  3. Eiffel Tower (CORRECT! but ranked 3rd) ✅
  4. Another building (wrong) ❌
  5. Yet another tower (wrong) ❌
```

In this case, the correct match is at position 3, so **Recall@1 = 0%** (we didn't get it in the top 1).

### What Phase 2 Provides

**Phase 2** performs **detailed image matching** between the query and each of the top-k predictions. It counts **inliers** (geometrically consistent keypoint matches).

**Inliers** = The number of keypoint matches that are geometrically consistent (validated through methods like RANSAC). More inliers typically means a better match because:
- Many consistent matches suggest the images show the same place
- Few matches suggest the images are different places

**Example Results from Phase 2**:
```
Query vs Prediction 1: 5 inliers (few matches, probably wrong)
Query vs Prediction 2: 3 inliers (very few, definitely wrong)
Query vs Prediction 3: 150 inliers (many matches, likely correct!) ✅
Query vs Prediction 4: 8 inliers (few matches, probably wrong)
Query vs Prediction 5: 12 inliers (some matches, but not great)
```

### What Reranking Does

**Reranking** takes the same top-k predictions and **reorders them by inlier count** (descending) instead of descriptor similarity.

**Before Reranking** (Phase 1 order - by descriptor similarity):
```
Position 1: Prediction 1 (5 inliers) ❌
Position 2: Prediction 2 (3 inliers) ❌
Position 3: Prediction 3 (150 inliers) ✅ CORRECT!
Position 4: Prediction 4 (8 inliers) ❌
Position 5: Prediction 5 (12 inliers) ❌
```

**After Reranking** (by inlier count - descending):
```
Position 1: Prediction 3 (150 inliers) ✅ CORRECT! (moved from 3→1)
Position 2: Prediction 5 (12 inliers) (moved from 5→2)
Position 3: Prediction 4 (8 inliers) (moved from 4→3)
Position 4: Prediction 1 (5 inliers) (moved from 1→4)
Position 5: Prediction 2 (3 inliers) (moved from 2→5)
```

### Why This Improves Results

The improvement is dramatic:

- **Before reranking**: Recall@1 = 0% (correct match at position 3)
- **After reranking**: Recall@1 = 100% (correct match at position 1) 🎉

**Key Insight**: Geometric consistency (measured by inliers) is often a **stronger signal** than global descriptor similarity. Two images of the same place will have many geometrically consistent matches, even if their global appearance is similar to other places.

### How It Works in Code

Looking at `reranking.py`, here's what happens:

1. **Line 46**: Gets geographic distances for each prediction (to check if it's correct)
2. **Lines 48-51**: Loads inlier counts from Phase 2 matching results
3. **Line 52**: **Sorts predictions by inlier count** (descending order)
4. **Line 53**: Reorders the geographic distances accordingly
5. **Lines 55-58**: Checks if any of the top-n reranked predictions are correct (within distance threshold)

The critical code is:
```python
# Sort by inlier count (descending)
query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
# Reorder predictions accordingly
geo_dists = geo_dists[indices]
```

This reorders the predictions so the one with the **most inliers is first**, which typically improves recall metrics significantly.

### Real-World Impact

In practice, reranking often improves:
- **Recall@1**: From ~60-70% to ~80-90% (depending on dataset)
- **Recall@5**: From ~85-90% to ~95-98%
- **Recall@10**: Smaller improvements, but still noticeable

The improvement is most dramatic for **Recall@1** because reranking helps push the correct match to the top position.

### Summary

- **Phase 1**: Gets top-k by descriptor similarity (fast, but not always accurate)
- **Phase 2**: Computes inlier counts for each top-k prediction (detailed matching)
- **Phase 3 (Reranking)**: Reorders top-k by inlier count → **better recall!**

Reranking is essentially saying: *"Instead of trusting global appearance similarity, let's trust geometric consistency - if two images have many matching keypoints, they're probably the same place."*

This is why reranking is a crucial step in improving VPR accuracy! 🚀

