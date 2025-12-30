# Galaxy Morphology and Photometric Redshifts

A machine learning system for galaxy morphological classification and photometric redshift estimation using Galaxy Zoo crowdsourced labels and SDSS photometric data.

## Overview

This project tackles two interconnected astronomical ML problems:
1. **Morphology Classification**: Classifying galaxies into morphological types (spiral, elliptical, irregular) using crowdsourced labels from Galaxy Zoo
2. **Photometric Redshift Estimation**: Predicting galaxy redshifts from broadband photometry (ugriz) without expensive spectroscopic observations

## Data Sources

- **Galaxy Zoo 2**: ~300,000 galaxies with crowdsourced morphological classifications
- **SDSS PhotoObj Table**: Photometric measurements (ugriz bands) for millions of galaxies
- **SDSS SpecObj Table**: Spectroscopic redshifts for training photo-z models (~1M examples)

## Project Structure

```
galaxy-morphology-photoz/
├── data/                    # Data storage (gitignored)
│   ├── raw/                 # Original downloaded data
│   ├── processed/           # Cleaned and cross-matched data
│   └── catalogs/            # Reference catalogs
├── src/                     # Source code
│   ├── data/                # Data loading, queries, cross-matching
│   ├── features/            # Feature engineering for photometry
│   ├── models/              # Model implementations
│   │   ├── morphology/      # Classification models
│   │   └── photoz/          # Redshift estimation models
│   └── visualization/       # Plotting utilities
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
└── configs/                 # Configuration files
```

## Key Features

- SDSS catalog querying via CasJobs/astroquery
- Cross-matching Galaxy Zoo with SDSS photometry
- Morphology classification (Random Forest, CNN on images)
- Photo-z estimation with uncertainty quantification
- Bias correction for spectroscopic selection effects

## Requirements

- Python 3.10+
- Astropy and astroquery
- Scikit-learn
- PyTorch (optional, for deep learning)
- Pandas, NumPy

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/galaxy-morphology-photoz.git
cd galaxy-morphology-photoz

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Galaxy Zoo data
python src/data/download_galaxy_zoo.py
```

## License

MIT License

## References

- Galaxy Zoo: https://www.galaxyzoo.org
- SDSS SkyServer: https://skyserver.sdss.org
- Astropy Documentation: https://docs.astropy.org
