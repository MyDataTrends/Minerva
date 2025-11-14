import argparse
import json
import pandas as pd
from alignment_drift_monitor import generate_historical_stats, monitor_alignment_drift


def main():
    parser = argparse.ArgumentParser(description="Alignment Drift Monitor")
    parser.add_argument("--baseline", required=True, help="Path to clean baseline CSV")
    parser.add_argument("--new", required=True, help="Path to new CSV batch")
    parser.add_argument("--threshold", type=float, default=0.2, help="Drift threshold")
    parser.add_argument("--block-on-drift", action="store_true", help="Exit with error if drift detected")
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline)
    stats = generate_historical_stats(baseline_df)

    new_df = pd.read_csv(args.new)
    result = monitor_alignment_drift(new_df, stats, drift_threshold=args.threshold, return_drifted_rows=True)

    print(json.dumps(result, indent=2))
    if result["drift_detected"] and args.block_on_drift:
        raise SystemExit("Alignment drift detected beyond threshold")


if __name__ == "__main__":
    main()
