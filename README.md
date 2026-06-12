# Physics-Informed-Machine-Learning-approach-for-estimating-FAPAR-from-Landsat-surface-reflectance

This repository provides two models, test code, and test data for the Physics-Informed Machine Learning approach for estimating spatiotemporally seamless FAPAR from Landsat surface reflectance.

1. **clear-sky modle**: estimate Physics-Informed FAPAR utilizing original Landsat observations
2. **cloudy-sky modle**: reconstructing spatiotemporally seamless FAPAR utilizing multi-year FAPAR combined with spatiotemporal information and environmental variables

## Repository structure

```text
├── LICENSE                 MIT license
├── README.md               Repository description, data format, and usage instructions
├── clear-sky model.py      Test code
├── cloudy-sky model.py     Test code
├── example clear-sky.zip   Test data   
├── example cloudy-sky.zip  Test data         
└── model.zip               Two models
```

## Software requirements

Python 3.9 or later is recommended. Required Python packages are:

```text
numpy
scipy
rasterio
lightgbm
scikit-learn
```

Install them using:

```bash
pip install numpy scipy rasterio lightgbm scikit-learn
```

## License

This code is released under the MIT License.

## Author

yimin6063-png
