"""
Grid point reduction code: GridPointsReduction.py

- K-means centroid extraction
- Train surrogate model
- Test performance
- Loop over number of centroids (NC)
"""

"""
Input features reduction code: InputFeaturesReduction.py
- Correlation matrix analysis
- Full vs reduced feature training
- Performance comparison
"""

"""
Storm set reduction (Adaptive sampling algorithm) code: StormSetReduction(AdaptiveSampling).py
- Loads training data
- Trains surrogate once
- Runs ONE adaptive-learning iteration
- Selects next storm
"""


%% Storm track & intensity visualization code: VisualizationOfStormSet.m
% This script demonstrates how to:
% 1) Plot storm tracks (lat/lon)
% 2) Compare intensity (Pc) distributions
% 3) Use different storm index sets
%
