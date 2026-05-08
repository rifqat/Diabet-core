"""
AutoSCAN: Automatic Detection of DBSCAN Parameters and Efficient
Clustering of Data in Overlapping Density Regions

Implementation based on:
  Bushra, A.A., Kim, D., Kan, Y. and Yi, G. (2024).
  AutoSCAN: automatic detection of DBSCAN parameters and efficient
  clustering of data in overlapping density regions.
  PeerJ Computer Science, 10, e1921.
  DOI: 10.7717/peerj-cs.1921

This module implements the two core contributions of AutoSCAN:
  1. Automatic epsilon detection via kNN frequency distribution + B-spline
  2. ClosestCore computation for accurate border object clustering
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.interpolate import make_interp_spline


class AutoSCAN:
    """
    AutoSCAN clustering with automatic epsilon detection and
    ClosestCore border-object reassignment.

    Parameters
    ----------
    min_samples : int or None
        Minimum number of samples in a neighbourhood to form a core point.
        If None, automatically set to 2 * number_of_features.
    """

    def __init__(self, min_samples=None):
        self.min_samples = min_samples
        self.epsilon_ = None
        self.labels_ = None

    def _compute_knn_distances(self, X):
        """
        Compute kNN distances for each point.
        k = min_samples (= 2 * d by default per AutoSCAN paper).
        Returns sorted kNN distances (ascending) for each point.
        """
        k = self.min_samples
        # k cannot exceed n_samples - 1
        k = min(k, X.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        # Return the k-th nearest neighbour distance for each point
        kth_distances = distances[:, -1]
        return kth_distances

    def _detect_epsilon_bspline(self, kth_distances):
        """
        Detect optimal epsilon using kNN frequency distribution and
        B-spline curve fitting.

        Steps (from AutoSCAN paper):
        1. Compute frequency distribution of kNN distances
        2. Fit cubic B-spline to smooth the distribution
        3. Iterate backwards through the smoothed curve to find the
           'valley' where the mean starts increasing
        4. Set epsilon to the mean of the steepest incline region
        """
        n_samples = len(kth_distances)

        # Number of bins for frequency distribution
        n_bins = min(max(30, n_samples // 10), 200)
        counts, bin_edges = np.histogram(kth_distances, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # B-spline smoothing
        n_spline_points = min(len(bin_centers) * 2, n_samples)
        if len(bin_centers) < 4:
            # Not enough points for cubic B-spline, fall back to median
            return float(np.median(kth_distances))

        try:
            # Cubic B-spline (k=3)
            spline = make_interp_spline(bin_centers, counts.astype(float), k=3)
            x_smooth = np.linspace(bin_centers[0], bin_centers[-1], n_spline_points)
            y_smooth = spline(x_smooth)
            # Ensure non-negative
            y_smooth = np.maximum(y_smooth, 0)
        except Exception:
            # Fallback: use raw histogram
            x_smooth = bin_centers
            y_smooth = counts.astype(float)

        # Window-based valley detection (iterate backwards)
        # Window size: ~10% of the curve length
        w = max(3, len(x_smooth) // 10)

        best_eps = float(np.median(kth_distances))  # default fallback
        found = False

        # Start from the end (maximum epsilon) and move backwards
        prev_mean = np.mean(y_smooth[-w:]) if len(y_smooth) >= w else np.mean(y_smooth)

        for i in range(len(x_smooth) - w - 1, -1, -1):
            window = y_smooth[i:i + w]
            curr_mean = np.mean(window)

            if curr_mean > prev_mean:
                # Mean started increasing — we found the valley
                # The optimal epsilon is the mean of the epsilon values in this region
                valley_start = i
                valley_end = min(i + w, len(x_smooth))
                best_eps = float(np.mean(x_smooth[valley_start:valley_end]))
                found = True
                break

            prev_mean = curr_mean

        if not found:
            # If no valley detected, use the "knee" of the sorted kNN distances
            sorted_dists = np.sort(kth_distances)
            # Use the point where the slope changes most (simplified knee)
            diffs = np.diff(sorted_dists)
            if len(diffs) > 0:
                knee_idx = np.argmax(diffs)
                best_eps = float(sorted_dists[knee_idx])

        return best_eps

    def _closest_core_dbscan(self, X, eps, min_samples):
        """
        Run DBSCAN with ClosestCore reassignment for border objects.

        The ClosestCore method (AutoSCAN's 2nd contribution):
        For each border object, find the closest core object and assign
        the border object to that core object's cluster.
        """
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        # Identify core, border, and noise points
        core_mask = np.zeros(len(X), dtype=bool)
        if hasattr(db, 'core_sample_indices_') and db.core_sample_indices_ is not None:
            core_mask[db.core_sample_indices_] = True

        core_indices = np.where(core_mask)[0]

        if len(core_indices) == 0:
            return labels

        # For border objects (not core, not noise), reassign to closest core
        # Border objects: assigned to a cluster but not core
        # In standard DBSCAN, border objects may be assigned inconsistently
        border_mask = (~core_mask) & (labels != -1)
        border_indices = np.where(border_mask)[0]

        if len(border_indices) == 0:
            return labels

        # Compute distances from border objects to all core objects
        core_points = X[core_indices]
        nn_core = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_core.fit(core_points)

        border_points = X[border_indices]
        distances, nearest_core_local = nn_core.kneighbors(border_points)

        # Reassign border objects to the cluster of their closest core object
        for i, border_idx in enumerate(border_indices):
            closest_core_idx = core_indices[nearest_core_local[i, 0]]
            labels[border_idx] = labels[closest_core_idx]

        return labels

    def fit_predict(self, X):
        """
        Run AutoSCAN on data X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Scaled input data.

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
            Cluster labels. -1 indicates noise/outlier.
        """
        n_features = X.shape[1]

        # Set minPts = 2 * d (as per AutoSCAN paper)
        if self.min_samples is None:
            self.min_samples = max(2, 2 * n_features)

        # Step 1: Compute kNN distances
        kth_distances = self._compute_knn_distances(X)

        # Step 2: Detect optimal epsilon via B-spline method
        self.epsilon_ = self._detect_epsilon_bspline(kth_distances)

        # Safety bounds
        self.epsilon_ = max(self.epsilon_, 1e-6)

        # Step 3: Run DBSCAN with ClosestCore reassignment
        self.labels_ = self._closest_core_dbscan(
            X, eps=self.epsilon_, min_samples=self.min_samples
        )

        return self.labels_

    def get_clean_mask(self):
        """
        Return a boolean mask where True = inlier (not noise).
        This enables using AutoSCAN as an outlier removal method
        comparable to YadroSeg.
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit_predict() first.")
        return self.labels_ != -1
