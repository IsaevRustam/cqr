import numpy as np
from cqr.real_data import load_rf1
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

X, y, info = load_rf1()
print(f"n={len(y)}, d={X.shape[1]}")
print(f"y stats: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}, std={y.std():.1f}")
print(f"         median={np.median(y):.1f}, Q1={np.percentile(y,25):.1f}, Q3={np.percentile(y,75):.1f}")
print(f"         IQR={np.percentile(y,75)-np.percentile(y,25):.1f}")
print(f"         skew={skew(y):.2f}, kurtosis={kurtosis(y):.2f}")

print("\nStandardScaler std vs RobustScaler IQR across 10 train splits:")
for seed in range(10):
    Xt, _, yt, _ = train_test_split(X, y, test_size=0.6, random_state=seed)
    ss = StandardScaler().fit(yt.reshape(-1, 1))
    rs = RobustScaler().fit(yt.reshape(-1, 1))
    print(f"  seed {seed}: SS scale={ss.scale_[0]:8.2f}  RS scale={rs.scale_[0]:8.2f}  "
          f"train range=[{yt.min():.1f}, {yt.max():.1f}]")
