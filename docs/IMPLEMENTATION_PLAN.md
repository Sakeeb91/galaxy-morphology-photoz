# Galaxy Morphology and Photometric Redshifts - Implementation Plan

## Expert Role

**ML Engineer with Astrophysics Domain Expertise**

This role is optimal because the project requires:
- Understanding of multi-class classification and regression (morphology + photo-z)
- Knowledge of astronomical photometry and survey selection effects
- Experience with crowdsourced label aggregation (Galaxy Zoo voting)
- Familiarity with database querying (SDSS CasJobs) and catalog cross-matching

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GALAXY MORPHOLOGY & PHOTO-Z PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Data Sources │───▶│ Cross-Match  │───▶│   Feature    │                   │
│  │              │    │   Engine     │    │  Extractor   │                   │
│  │ - Galaxy Zoo │    │              │    │              │                   │
│  │ - SDSS Photo │    │ - objID join │    │ - Colors     │                   │
│  │ - SDSS Spec  │    │ - Coord xmatch│   │ - Magnitudes │                   │
│  └──────────────┘    └──────────────┘    │ - Petro/PSF  │                   │
│                                          └──────┬───────┘                   │
│                                                 │                           │
│                    ┌────────────────────────────┴───────────────────────┐   │
│                    │                                                     │   │
│                    ▼                                                     ▼   │
│  ┌─────────────────────────┐                     ┌─────────────────────────┐│
│  │  MORPHOLOGY CLASSIFIER  │                     │    PHOTO-Z ESTIMATOR    ││
│  │                         │                     │                         ││
│  │  ┌─────────────────┐    │                     │  ┌─────────────────┐    ││
│  │  │ Label Processor │    │                     │  │  Bias Corrector │    ││
│  │  │ (vote fractions)│    │                     │  │ (selection func)│    ││
│  │  └────────┬────────┘    │                     │  └────────┬────────┘    ││
│  │           ▼             │                     │           ▼             ││
│  │  ┌─────────────────┐    │                     │  ┌─────────────────┐    ││
│  │  │   RF / XGBoost  │    │                     │  │   RF / GPR      │    ││
│  │  │  Multi-class    │    │                     │  │  Regression     │    ││
│  │  └────────┬────────┘    │                     │  └────────┬────────┘    ││
│  │           ▼             │                     │           ▼             ││
│  │  ┌─────────────────┐    │                     │  ┌─────────────────┐    ││
│  │  │ Spiral/Ellip/   │    │                     │  │ z_phot + sigma  │    ││
│  │  │ Irregular/Merge │    │                     │  │ (with uncert.)  │    ││
│  │  └─────────────────┘    │                     │  └─────────────────┘    ││
│  └─────────────────────────┘                     └─────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: Download Galaxy Zoo labels and query SDSS photometry
2. **Cross-matching**: Join datasets on objID or sky coordinates
3. **Feature Engineering**: Compute colors, color gradients, concentration indices
4. **Morphology Path**: Train classifier on vote-fraction labels
5. **Photo-z Path**: Train regressor on spectroscopic redshifts, correct for selection
6. **Inference**: Predict morphology class and redshift with uncertainties

---

## Technology Selection

| Component | Choice | Rationale | Fallback |
|-----------|--------|-----------|----------|
| SDSS Queries | astroquery | Programmatic access | CasJobs web |
| Cross-matching | Astropy coordinates | Sub-arcsec matching | Pandas merge |
| Morphology Model | XGBoost | Handles multi-class, feature importance | Random Forest |
| Photo-z Model | Random Forest | Fast, uncertainty via variance | GPR (slow but probabilistic) |
| Uncertainty | Quantile regression | Distribution of predictions | Bootstrap |
| Visualization | Matplotlib + Seaborn | Publication quality | Plotly |
| Testing | Pytest | Standard | Unittest |

### Tradeoffs

- **Tabular vs Image-based**: Start with photometry features. Image CNN only if tabular <0.9 accuracy.
- **XGBoost vs RF**: XGBoost is faster with similar accuracy. Use RF for easier uncertainty.
- **Point vs PDF photo-z**: Start with point estimates + sigma. Full PDF if needed for science.

---

## Phased Implementation Plan

### Phase 1: Data Acquisition and Cross-Matching

**Scope**: Download all required data and create unified catalog

**Files to Create**:
- `src/data/__init__.py`
- `src/data/download_galaxy_zoo.py` - GZ download
- `src/data/query_sdss.py` - SDSS photometry queries
- `src/data/cross_match.py` - Catalog joining
- `configs/data_config.yaml` - Configuration

**Deliverables**:
- Galaxy Zoo labels downloaded (~300K galaxies)
- SDSS photometry for matched objects
- Spectroscopic redshifts for ~1M galaxies
- Unified catalog with morphology labels, photometry, and spec-z where available

**Verification**:
```bash
python src/data/download_galaxy_zoo.py
python src/data/query_sdss.py --sample 10000  # Start small
python src/data/cross_match.py --gz data/raw/gz2.csv --sdss data/raw/sdss_photo.csv
# Should produce: data/processed/unified_catalog.csv
```

**Technical Challenges**:
- SDSS CasJobs has query limits (split into chunks)
- Galaxy Zoo has multiple classification schemes (GZ1 vs GZ2)
- Coordinate cross-matching requires proper error handling

**Definition of Done**:
- [ ] Galaxy Zoo 2 morphology labels downloaded
- [ ] SDSS photometry for matched objIDs
- [ ] Spectroscopic subset identified for photo-z training
- [ ] Unified catalog with no duplicate objIDs

**Code Skeleton**:
```python
# src/data/download_galaxy_zoo.py
"""Download Galaxy Zoo 2 morphological classifications."""

from pathlib import Path
import pandas as pd


def download_galaxy_zoo_2(output_dir: Path) -> Path:
    """Download Galaxy Zoo 2 data from Kaggle.

    Args:
        output_dir: Where to save the data

    Returns:
        Path to downloaded CSV file
    """
    # TODO: Implement Kaggle download or direct download
    pass


def process_vote_fractions(gz_data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw votes to morphology probabilities.

    Galaxy Zoo provides vote counts. We need:
    - P(spiral): smooth_or_featured = 'featured' AND has_spiral_arms
    - P(elliptical): smooth_or_featured = 'smooth'
    - P(irregular/merger): from merger and irregular columns

    Args:
        gz_data: Raw Galaxy Zoo DataFrame

    Returns:
        DataFrame with morphology probabilities
    """
    # TODO: Implement vote aggregation
    pass
```

---

### Phase 2: Feature Engineering

**Scope**: Create photometric features from SDSS measurements

**Files to Create**:
- `src/features/__init__.py`
- `src/features/colors.py` - Color calculations
- `src/features/morphometric.py` - Concentration, asymmetry
- `src/features/feature_pipeline.py` - Combined pipeline

**Deliverables**:
- Color features (u-g, g-r, r-i, i-z)
- Magnitude features (Petrosian, PSF, model)
- Morphometric features (concentration, Petrosian radius ratio)
- Feature matrix for all galaxies

**Key Features**:
1. **Colors** (redshift-sensitive):
   - u-g, g-r, r-i, i-z (adjacent bands)
   - u-r, g-i (wider baselines)

2. **Magnitudes** (brightness):
   - r-band Petrosian (total flux)
   - PSF magnitude (point-like contribution)
   - Model magnitude (best-fit Sersic)

3. **Morphometric** (shape):
   - Concentration index: C = R90/R50
   - Petrosian half-light radius
   - Axis ratio (b/a)
   - Fracdev (de Vaucouleurs fraction)

**Verification**:
```bash
python -c "from src.features import compute_all_features; print(compute_all_features(test_catalog).columns)"
# Should output: ['u_g', 'g_r', 'r_i', 'i_z', 'petro_r', 'concentration', ...]
```

**Technical Challenges**:
- Magnitude errors propagate to color errors
- Missing values in some bands (non-detections)
- Galactic extinction corrections (use SDSS dereddened mags)

**Definition of Done**:
- [ ] All color features computed
- [ ] Morphometric features extracted
- [ ] Missing value strategy documented
- [ ] Feature correlation matrix visualized

---

### Phase 3: Morphology Classification

**Scope**: Train multi-class classifier for galaxy morphology

**Files to Create**:
- `src/models/__init__.py`
- `src/models/morphology/__init__.py`
- `src/models/morphology/classifier.py` - XGBoost wrapper
- `src/models/morphology/train.py` - Training script

**Deliverables**:
- Trained XGBoost classifier
- Multi-class accuracy >85%
- Confusion matrix across morphology types
- Feature importance ranking

**Label Processing**:
- Galaxy Zoo provides vote fractions, not hard labels
- Option 1: Use majority vote (>50% agreement)
- Option 2: Weight samples by vote confidence
- Option 3: Multi-label with soft targets

**Verification**:
```bash
python src/models/morphology/train.py --output models/morphology_xgb.pkl
python src/models/morphology/evaluate.py --model models/morphology_xgb.pkl
# Should print accuracy, confusion matrix, classification report
```

**Technical Challenges**:
- Label noise from crowdsourced votes
- Class imbalance (spirals more common than mergers)
- Morphology depends on viewing angle (edge-on vs face-on)

**Definition of Done**:
- [ ] Classifier achieves >85% accuracy on majority-vote labels
- [ ] Per-class precision/recall documented
- [ ] Top 10 important features identified
- [ ] Model saved and reproducibly loadable

---

### Phase 4: Photometric Redshift Estimation

**Scope**: Train regression model for photo-z

**Files to Create**:
- `src/models/photoz/__init__.py`
- `src/models/photoz/regressor.py` - RF regressor with uncertainty
- `src/models/photoz/train.py` - Training script
- `src/models/photoz/bias_correction.py` - Selection effect correction

**Deliverables**:
- Trained photo-z regressor
- RMSE < 0.03 in (z_phot - z_spec) / (1 + z_spec)
- Outlier fraction (|delta_z| > 0.15) < 5%
- Per-galaxy uncertainty estimates

**Verification**:
```bash
python src/models/photoz/train.py --output models/photoz_rf.pkl
python src/models/photoz/evaluate.py --model models/photoz_rf.pkl
# Should print: sigma_NMAD, outlier_fraction, bias
```

**Technical Challenges**:
- Spectroscopic selection bias (brighter galaxies overrepresented)
- Redshift degeneracies (different galaxy types at different z look similar)
- Uncertainty calibration (are 1-sigma errors actually 68%?)

**Selection Bias Mitigation**:
- Weight training samples by inverse selection probability
- Or train on magnitude-limited subset and document validity range

**Definition of Done**:
- [ ] sigma_NMAD < 0.03
- [ ] Outlier fraction < 5%
- [ ] Uncertainty calibration verified (pit histogram)
- [ ] Selection bias documented

---

### Phase 5: Production Pipeline

**Scope**: End-to-end inference on new galaxies

**Files to Create**:
- `src/predict.py` - Unified inference script
- `src/visualization/galaxy_viz.py` - Result visualization

**Deliverables**:
- CLI for processing new SDSS objIDs
- Combined output: morphology + photo-z + uncertainties
- SDSS image thumbnail with overlaid info

**Definition of Done**:
- [ ] Can query SDSS by objID and return morphology + redshift
- [ ] Output includes uncertainty estimates
- [ ] Visualization shows galaxy image with predictions

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning | Mitigation |
|------|------------|--------|---------------|------------|
| SDSS query rate limiting | High | Low | 429 errors | Add retry logic, cache results |
| GZ vote noise hurts accuracy | Medium | Medium | Accuracy <0.8 | Use vote confidence weighting |
| Spec-z selection bias | High | High | Photo-z biased at faint end | Weight by selection function |
| Cross-match failures | Medium | Medium | Unmatched fraction >10% | Widen match radius, check coords |
| Color errors dominate | Low | Medium | Color scatter too high | Use dereddened mags, clip outliers |

---

## Testing Strategy

### Testing Framework: Pytest

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test color calculations on known values
   - Test cross-matching with synthetic catalogs
   - Test feature pipeline output shapes

2. **Integration Tests** (`tests/integration/`)
   - Test full pipeline from objID to prediction
   - Test with known galaxies (should match literature)

3. **Validation Tests** (`tests/validation/`)
   - Test model metrics against benchmarks
   - Test uncertainty calibration

### First Three Tests to Write

```python
# tests/unit/test_colors.py
import numpy as np
import pandas as pd
from src.features.colors import compute_colors

def test_color_calculation():
    """Color should be difference of magnitudes."""
    data = pd.DataFrame({
        'u': [20.0],
        'g': [19.5],
        'r': [19.0]
    })
    colors = compute_colors(data)
    assert np.isclose(colors['u_g'].iloc[0], 0.5)
    assert np.isclose(colors['g_r'].iloc[0], 0.5)

def test_missing_mag_handling():
    """Missing magnitudes should produce NaN colors."""
    data = pd.DataFrame({
        'u': [np.nan],
        'g': [19.5],
        'r': [19.0]
    })
    colors = compute_colors(data)
    assert np.isnan(colors['u_g'].iloc[0])
    assert np.isclose(colors['g_r'].iloc[0], 0.5)

def test_cross_match_unique():
    """Cross-matched catalog should have unique objIDs."""
    from src.data.cross_match import cross_match_catalogs
    gz = pd.DataFrame({'objID': [1, 2, 3], 'morph': ['S', 'E', 'S']})
    sdss = pd.DataFrame({'objID': [1, 2, 2, 3], 'mag_r': [18, 19, 19.1, 20]})
    matched = cross_match_catalogs(gz, sdss, on='objID')
    assert len(matched) == len(matched['objID'].unique())
```

---

## First Concrete Task

### File to Create: `src/data/download_galaxy_zoo.py`

### Function Signature:
```python
def download_galaxy_zoo_2(
    output_dir: str = "data/raw",
    source: str = "kaggle"
) -> Path:
    """Download Galaxy Zoo 2 morphological classifications.

    Args:
        output_dir: Directory to save downloaded files
        source: 'kaggle' or 'zenodo'

    Returns:
        Path to the downloaded CSV file

    Raises:
        ValueError: If source not recognized
        FileNotFoundError: If Kaggle credentials not configured
    """
```

### Starter Code:
```python
# src/data/download_galaxy_zoo.py
"""Download Galaxy Zoo 2 morphological classifications."""

import os
from pathlib import Path
import subprocess


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_galaxy_zoo_2(
    output_dir: str = "data/raw",
    source: str = "kaggle"
) -> Path:
    """Download Galaxy Zoo 2 morphological classifications.

    Args:
        output_dir: Directory to save downloaded files
        source: 'kaggle' for Kaggle datasets

    Returns:
        Path to the downloaded data directory

    Raises:
        ValueError: If source not recognized
        FileNotFoundError: If credentials not configured
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if source == "kaggle":
        if not check_kaggle_credentials():
            raise FileNotFoundError(
                "Kaggle credentials not found. Place kaggle.json in ~/.kaggle/"
            )

        # Galaxy Zoo 2 on Kaggle
        dataset = "pavansanagapati/galaxy-zoo-the-galaxy-challenge"

        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset,
            "-p", str(output_path),
            "--unzip"
        ]

        print(f"Downloading Galaxy Zoo 2 from Kaggle...")
        subprocess.run(cmd, check=True)
        print(f"Downloaded to {output_path}")

    else:
        raise ValueError(f"Unknown source: {source}. Use 'kaggle'.")

    return output_path


def load_galaxy_zoo_labels(data_dir: Path) -> "pd.DataFrame":
    """Load and parse Galaxy Zoo 2 labels.

    Args:
        data_dir: Directory containing downloaded files

    Returns:
        DataFrame with objID and morphology vote fractions
    """
    import pandas as pd

    # The Galaxy Zoo challenge data includes training_solutions.csv
    labels_file = data_dir / "training_solutions_rev1.csv"

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    df = pd.read_csv(labels_file)
    print(f"Loaded {len(df)} galaxies with morphology labels")

    return df


if __name__ == "__main__":
    data_dir = download_galaxy_zoo_2()
    labels = load_galaxy_zoo_labels(data_dir)
    print(labels.head())
```

### Verification:
```bash
# Ensure kaggle is installed
pip install kaggle

# Run the download script
python src/data/download_galaxy_zoo.py

# Verify files exist
ls data/raw/
# Should show Galaxy Zoo files
```

### First Commit Message:
```
feat: Add Galaxy Zoo 2 download utility

- Implement download_galaxy_zoo_2() for fetching morphology labels
- Add Kaggle credential checking
- Include label loading function for training_solutions.csv
```

---

## Learning Notes (for Junior Developer)

### Concepts to Understand Before Coding:

1. **Galaxy Morphology**: Galaxies are classified by shape - spirals (disk + arms), ellipticals (smooth, round), irregulars (no clear structure), mergers (two galaxies colliding).

2. **Photometric Redshift**: Redshift (z) measures how far away a galaxy is. Spectroscopy gives precise z but is expensive. We estimate z from colors because galaxy spectra shift predictably with distance.

3. **SDSS Magnitude System**: SDSS uses 5 filters (ugriz) spanning UV to near-IR. Magnitudes are logarithmic brightness; colors are magnitude differences.

4. **Selection Effects**: Spectroscopic surveys are biased toward bright galaxies. Training on spec-z data means our model may fail on faint galaxies.

5. **Crowdsourced Labels**: Galaxy Zoo aggregates votes from many volunteers. Vote fractions represent uncertainty - a 60-40 split means ambiguous morphology.

### Resources:
- Galaxy Zoo science: https://www.galaxyzoo.org/about
- SDSS photometry: https://www.sdss.org/dr16/imaging/
- Photo-z review: https://arxiv.org/abs/1903.02016
- Astropy tutorials: https://learn.astropy.org/
