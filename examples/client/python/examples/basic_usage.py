"""Basic usage example for the Hyprstream client."""

import time
from hyprstream_client import MetricsClient, MetricRecord

def main():
    """Example usage demonstrating the client's features."""
    with MetricsClient() as client:
        # Example 1: Single metric insertion
        metric = MetricRecord(
            metric_id="test_metric_1",
            timestamp=int(time.time() * 1e9),
            value_running_window_sum=10.0,
            value_running_window_avg=2.0,
            value_running_window_count=5
        )
        client.set_metric(metric)
        
        # Example 2: Batch insertion
        batch_metrics = [
            MetricRecord(
                metric_id=f"test_metric_{i}",
                timestamp=int(time.time() * 1e9),
                value_running_window_sum=float(i * 10),
                value_running_window_avg=float(i),
                value_running_window_count=10
            )
            for i in range(2, 5)
        ]
        client.set_metrics_batch(batch_metrics)
        
        # Example 3: Query with time window
        print("\nLast minute of metrics:")
        df = client.get_metrics_window(60)
        print(df)
        
        # Example 4: Query with specific metric IDs
        print("\nSpecific metrics:")
        df = client.query_metrics(
            metric_ids=["test_metric_1", "test_metric_2"],
            from_timestamp=int((time.time() - 3600) * 1e9)  # Last hour
        )
        print(df)

if __name__ == "__main__":
    main() 