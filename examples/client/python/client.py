import pyarrow as pa
import pyarrow.flight as flight
import adbc_driver_flightsql
import adbc_driver_flightsql.dbapi
import pandas as pd
import time
from typing import List, Optional, Dict, Any

class MetricsClient:
    def __init__(self, connection_string: str = "grpc://localhost:50051"):
        """Initialize the metrics client with a connection string."""
        self.connection_string = connection_string
        self.conn = None
        
    def connect(self):
        """Establish connection to the Flight SQL server."""
        print("Connecting to Flight SQL server")
        self.conn = adbc_driver_flightsql.dbapi.connect(self.connection_string)
        
    def disconnect(self):
        """Close the connection to the Flight SQL server."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def set_metric(self, metric_id: int, timestamp: Optional[int] = None,
                   value_sum: float = 0.0, value_avg: float = 0.0, 
                   value_count: int = 0) -> None:
        """Insert or update a single metric."""
        if not self.conn:
            raise ConnectionError("Not connected to server")
            
        if timestamp is None:
            timestamp = int(time.time() * 1e9)
            
        cursor = self.conn.cursor()
        try:
            query = """
            INSERT INTO metrics (metric_id, timestamp, valueRunningWindowSum, 
                               valueRunningWindowAvg, valueRunningWindowCount)
            VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(query, [metric_id, timestamp, value_sum, value_avg, value_count])
        finally:
            cursor.close()

    def set_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Insert multiple metrics at once."""
        if not self.conn:
            raise ConnectionError("Not connected to server")
            
        df = pd.DataFrame(metrics)
        if 'timestamp' not in df.columns:
            df['timestamp'] = int(time.time() * 1e9)
            
        cursor = self.conn.cursor()
        try:
            for _, row in df.iterrows():
                query = """
                INSERT INTO metrics (metric_id, timestamp, valueRunningWindowSum, 
                                   valueRunningWindowAvg, valueRunningWindowCount)
                VALUES (?, ?, ?, ?, ?)
                """
                cursor.execute(query, [
                    row['metric_id'],
                    row['timestamp'],
                    row['valueRunningWindowSum'],
                    row['valueRunningWindowAvg'],
                    row['valueRunningWindowCount']
                ])
        finally:
            cursor.close()
        print(f"Inserted {len(df)} metrics")

    def get_metrics(self, time_window_seconds: int = 60) -> pd.DataFrame:
        """Get metrics within the specified time window."""
        if not self.conn:
            raise ConnectionError("Not connected to server")
            
        current_time = int(time.time() * 1e9)
        start_time = current_time - (time_window_seconds * 1e9)
        
        cursor = self.conn.cursor()
        try:
            query = """
            SELECT * FROM metrics 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            LIMIT 100
            """
            cursor.execute(query, [start_time])
            results = cursor.fetch_arrow_table()
            
            if results.num_rows > 0:
                return results.to_pandas()
            return pd.DataFrame()
        finally:
            cursor.close()

    def update_metric(self, metric_id: int, value_sum: float, 
                     value_avg: float, value_count: int) -> None:
        """Update an existing metric by ID."""
        if not self.conn:
            raise ConnectionError("Not connected to server")
            
        cursor = self.conn.cursor()
        try:
            query = """
            UPDATE metrics 
            SET valueRunningWindowSum = ?, 
                valueRunningWindowAvg = ?, 
                valueRunningWindowCount = ?
            WHERE metric_id = ?
            """
            cursor.execute(query, [value_sum, value_avg, value_count, metric_id])
        finally:
            cursor.close()
        print(f"Updated metric_id {metric_id}")

    def delete_metric(self, metric_id: int) -> None:
        """Delete a metric by ID."""
        if not self.conn:
            raise ConnectionError("Not connected to server")
            
        cursor = self.conn.cursor()
        try:
            query = "DELETE FROM metrics WHERE metric_id = ?"
            cursor.execute(query, [metric_id])
        finally:
            cursor.close()
        print(f"Deleted metric_id {metric_id}")

def main():
    # Example usage
    with MetricsClient() as client:
        # Insert test data
        test_metrics = [
            {
                'metric_id': 1,
                'valueRunningWindowSum': 10.0,
                'valueRunningWindowAvg': 1.0,
                'valueRunningWindowCount': 10
            },
            {
                'metric_id': 2,
                'valueRunningWindowSum': 20.0,
                'valueRunningWindowAvg': 2.0,
                'valueRunningWindowCount': 10
            },
            {
                'metric_id': 3,
                'valueRunningWindowSum': 30.0,
                'valueRunningWindowAvg': 3.0,
                'valueRunningWindowCount': 10
            }
        ]
        client.set_metrics(test_metrics)
        
        # Query metrics
        df = client.get_metrics(time_window_seconds=60)
        print("\nQuery results:")
        print(df)
        
        # Update a metric
        client.update_metric(metric_id=1, value_sum=100.0, value_avg=10.0, value_count=10)
        
        # Delete a metric
        client.delete_metric(metric_id=2)

if __name__ == "__main__":
    main()