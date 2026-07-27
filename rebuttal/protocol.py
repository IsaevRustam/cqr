"""Frozen constants for exploratory and confirmatory rebuttal runs."""

EXPLORATORY_PROTOCOL = "exploratory"
CONFIRMATORY_PROTOCOL = "confirmatory"
PROTOCOL_CHOICES = (EXPLORATORY_PROTOCOL, CONFIRMATORY_PROTOCOL)

CONFIRMATORY_VERSION = "2026-07-27-v1"
CONFIRMATORY_BANDWIDTH = 1.4
CONFIRMATORY_SEEDS = list(range(142, 162))
CONFIRMATORY_METHOD_KEYS = ("global", "local_fixed_1.4")

PRIMARY_WGC_GROUPING = "base_interval_width"
PRIMARY_WGC_BINNING = "rank"
PRIMARY_WGC_BINS = 5
SENSITIVITY_WGC_BINS = (3, 5, 10)

# Reporting diagnostic only: it does not select h, remove points, or change
# the prediction interval.  See CONFIRMATORY_PROTOCOL.md.
ESS_REPORT_THRESHOLD = 30.0
