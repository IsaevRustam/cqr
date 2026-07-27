"""Frozen constants for exploratory and confirmatory rebuttal runs."""

EXPLORATORY_PROTOCOL = "exploratory"
CONFIRMATORY_PROTOCOL = "confirmatory"
SELECTED_PROTOCOL = "train_selected"
PROTOCOL_CHOICES = (EXPLORATORY_PROTOCOL, CONFIRMATORY_PROTOCOL, SELECTED_PROTOCOL)

CONFIRMATORY_VERSION = "2026-07-27-v1"
CONFIRMATORY_BANDWIDTH = 1.4
CONFIRMATORY_SEEDS = list(range(142, 162))
CONFIRMATORY_METHOD_KEYS = ("global", "local_fixed_1.4")

# Train-only bandwidth selection (protocol 'train_selected'): h is chosen by
# an inner 70/15/15 split of the TRAINING set (T-fit / T-cal / T-eval) and
# frozen before the real calibration set or the test set is touched.
# See rebuttal/h_selection.py and TRAIN_SELECTED_PROTOCOL.md.
# v1: candidates = silverman/scott/isj + the 9 fixed grid values (12-grid).
# v2: fixed grid only — the data-driven rules were dropped after the v1 runs
#     on six datasets showed they occasionally select degenerate small h
#     (median ESS < 10) that loses coverage. See the amendment note in
#     TRAIN_SELECTED_PROTOCOL.md.
SELECTED_VERSION = "2026-07-27-v2"
SELECTED_METHOD_KEYS = ("global", "local_selected")
SELECTED_SEEDS = list(range(142, 162))
INNER_SPLIT_FRACS = (0.70, 0.15, 0.15)

PRIMARY_WGC_GROUPING = "base_interval_width"
PRIMARY_WGC_BINNING = "rank"
PRIMARY_WGC_BINS = 5
SENSITIVITY_WGC_BINS = (3, 5, 10)

# Reporting diagnostic only: it does not select h, remove points, or change
# the prediction interval.  See CONFIRMATORY_PROTOCOL.md.
ESS_REPORT_THRESHOLD = 30.0
