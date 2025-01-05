# Hyprstream: Real-time Aggregation Windows and High-Performance Cache for Apache Arrow Flight SQL 🚀

Hyprstream is a next-generation application for real-time data ingestion, windowed aggregation, caching, and serving. Built on Apache Arrow Flight and DuckDB, and developed in Rust, Hyprstream dynamically calculates metrics like running sums, counts, and averages, enabling blazing-fast data workflows, intelligent caching, and seamless integration with ADBC-compliant datastores. Its real-time aggregation capabilities empower AI/ML pipelines and analytics with instant insights. 💾✨

## Key Features 🎯

### 🔄 Data Ingestion via Apache Arrow Flight

- **Streamlined Ingestion**: Ingests data efficiently using **Arrow Flight**, an advanced columnar data transport protocol
- **Real-Time Streaming**: Supports real-time metrics, datasets, and vectorized data for analytics and AI/ML workflows
- **Write-Through to ADBC**: Ensures data consistency with immediate caching and write-through to backend datastores

### 🧠 Intelligent Read Caching with DuckDB

- **In-Memory Performance**: Uses **DuckDB** for lightning-fast caching of frequently accessed data
- **Optimized Querying**: Stores query results and intermediate computations for analytics workloads
- **Automatic Management**: Handles caching transparently with configurable expiry policies

### ⚡ Real-Time Aggregation

- **Dynamic Metrics**: Maintains running sums, counts, and averages for real-time insights
- **Time Window Partitioning**: Supports fixed time windows (e.g., 5m, 30m, hourly, daily) for granular analysis
- **Lightweight State**: Maintains only aggregate states for efficient memory usage

### 🌐 Data Serving with Arrow Flight SQL

- **High-Performance Queries**: Serves cached data via Arrow Flight SQL for minimal latency
- **Vectorized Data**: Optimized for AI/ML pipelines and analytical queries
- **Seamless Integration**: Connects with analytics and visualization tools

## Benefits 🌟

- **🚀 Low Latency**: Millisecond-level query responses for cached data
- **⚙️ Scalable**: Handles large-scale data workflows with ease
- **🔗 Flexible**: Integrates with Postgres, Redis, Snowflake, and other ADBC datastores
- **🤖 AI/ML Ready**: Optimized for vectorized data and inference pipelines
- **📈 Real-Time Metrics**: Dynamic calculation of statistical metrics
- **⌛ Time Windows**: Granular control of metrics with configurable windows
- **⛭ Rust-Powered**: High-performance, memory-safe implementation

## Getting Started 🚀

1. Install Hyprstream:
   ```bash
   cargo install hyprstream
   ```

2. Start the server with default configuration:
   ```bash
   hyprstream
   ```

3. Use with custom configuration:
   ```bash
   hyprstream --config /path/to/config.toml
   ```

For configuration options and detailed documentation, run:
```bash
hyprstream --help
```

Or visit our [API Documentation](https://docs.rs/hyprstream) for comprehensive guides and examples.

## Example Usage 💡

### Quick Start with ADBC

Hyprstream implements the Arrow Flight SQL protocol, making it compatible with any ADBC-compliant client:

```python
import adbc_driver_flightsql.dbapi

# Connect to Hyprstream using standard ADBC
conn = adbc_driver_flightsql.dbapi.connect("grpc://localhost:50051")

try:
    cursor = conn.cursor()
    
    # Query metrics with time windows
    cursor.execute("""
        SELECT 
            metric_id,
            COUNT(*) as samples,
            AVG(value_running_window_avg) as avg_value
        FROM metrics
        WHERE timestamp >= NOW() - INTERVAL '1 hour'
        GROUP BY metric_id
        ORDER BY avg_value DESC
    """)
    
    results = cursor.fetch_arrow_table()
    print(results.to_pandas())
    
finally:
    cursor.close()
    conn.close()
```

### Using as a Cache Layer

Hyprstream can act as a high-performance cache in front of any ADBC-compliant database:

```bash
# Start with PostgreSQL backend and DuckDB cache
hyprstream \
  --storage-backend cached \
  --cache-backend duckdb \
  --cache-duration 3600 \
  --driver-path /usr/local/lib/libadbc_driver_postgresql.so \
  --db-url postgresql://localhost:5432 \
  --db-name metrics
```

Queries automatically use the cache:
```python
import adbc_driver_flightsql.dbapi

with adbc_driver_flightsql.dbapi.connect("grpc://localhost:50051") as conn:
    cursor = conn.cursor()
    
    # First query fetches from PostgreSQL and caches
    cursor.execute("SELECT * FROM large_table WHERE region = ?", ["us-west"])
    results = cursor.fetch_arrow_table()
```

For more examples and detailed documentation:
- [Python Client Examples](examples/client/python)
- [API Documentation](https://docs.rs/hyprstream)
- Run `hyprstream --help` for configuration options

## Better Together: Ecosystem Integration 🔄

Hyprstream enhances modern data architectures by filling critical gaps in the real-time data stack. While tools like Flink excel at complex stream processing and MotherDuck provides scalable cloud analytics, Hyprstream adds the missing piece: instant, SQL-based access to streaming data and real-time metrics. By supporting any ADBC-compliant database as a backend, including MotherDuck, Hyprstream enables architectures that combine cloud-scale storage with edge performance.

### Comparison with Stream Processing & Analytics Tools

| Feature | Hyprstream | Apache Flink | MotherDuck |
|---------|------------|--------------|------------|
| **Ingest-to-Query Latency** | 1-10ms* | Seconds-minutes** | 100ms-seconds |
| **Query Interface** | Direct SQL | External sink required | Direct SQL |
| **Storage Model** | In-memory + ADBC | External systems | Cloud-native |
| **Deployment** | Single binary | Cluster + job manager | Cloud service |
| **Scale Focus** | Hot data, edge | Stream processing | Cloud analytics |
| **State Management** | Time windows, metrics | Full event state | Full dataset |
| **Data Access** | Arrow Flight SQL | Custom operators | DuckDB/SQL |
| **Cost Model** | Compute-focused | Compute-focused | Storage-focused |

\* *For cached data; backend queries add typical ADBC database latency*
\** *End-to-end latency including writing to external storage and querying*

## Contributing 🤝

We welcome contributions! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
For inquiries or support, contact us at [support@hyprstream.com](mailto:support@hyprstream.com) or visit our GitHub repository to contribute! 🌐