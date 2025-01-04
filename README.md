# Hyprstream: Real-time Aggregation Windows and High-Performance Cache for Apache Arrow Flight SQL 🚀

Hyprstream is a next-generation application for real-time data ingestion, windowed aggregation, caching, and serving. Built on Apache Arrow Flight and DuckDB, and developed in Rust, Hyprstream dynamically calculates metrics like running sums, counts, and averages, enabling blazing-fast data workflows, intelligent caching, and seamless integration with ADBC-compliant datastores. Its real-time aggregation capabilities empower AI/ML pipelines and analytics with instant insights. 💾✨

## Key Features 🎯

### 🔄 Data Ingestion via Apache Arrow Flight

- **Streamlined Ingestion**: Ingests data efficiently using **Arrow Flight**, an advanced columnar data transport protocol.
- **Real-Time Streaming**: Supports real-time metrics, datasets, and vectorized data, making it perfect for analytics and AI/ML workflows.
- **Seamless Integration**: Works effortlessly with data producers for high-throughput ingestion.
- **Write-Through to ADBC Datastores**: Ensures eventual data consistency with immediate caching and write-through to backend datastores such as Postgres, Redis, and Snowflake via ADBC.

### 🧠 Intelligent Read Caching with DuckDB

- **In-Memory Performance**: Uses **DuckDB** for lightning-fast caching of frequently accessed data.
- **Optimized Querying**: Stores query results and intermediate computations for analytics workloads.
- **Automatic Management**: Handles caching transparently, reducing manual effort.
- **Expiry Policies**:
  - **Time-Based**: Automatically expires data after a configurable duration.
  - **LRU/LFU (Work in Progress)**: Future enhancements for intelligent cache eviction.

### 🌐 Data Serving with Arrow Flight SQL

- **High-Performance Queries**: Serves cached data via **Arrow Flight SQL** for minimal latency.
- **Supports Advanced Workflows**:
  - Vectorized data for AI/ML pipelines.
  - Analytical queries for real-time insights.
- **Downstream Integration**: Connects seamlessly with analytics and visualization tools.

### ⚡ Real-Time Aggregation
- **Dynamic Metrics**: Maintains running sums, counts, and averages for real-time insights.
- **Lightweight State Management**: Avoids storing the full dataset by maintaining only aggregate states.
- **Dynamic Weight Computation**: Enables the calculation of weights and biases in real time for AI/ML pipelines.
- **Instant Feedback**: Provides immediate access to derived metrics for analytics and inference.
- **Time Window Partitioning**: Supports partitioning aggregate calculations into fixed time windows for granular analysis.

## Application Workflow 🔧

### 1️⃣ Data Ingestion

1. Data producers send data to Hyprstream via the **Arrow Flight ingestion API**.
2. The data is processed and stored in the in-memory **DuckDB cache**.
3. Aggregates for sums, counts, and averages are dynamically updated as new data arrives.
4. Aggregate calculations can be partitioned into **time windows** (e.g., 5m, 30m, hourly, daily, weekly) for advanced insights.

### 2️⃣ Query Processing

1. Consumers send queries via **Arrow Flight SQL**.
2. Hyprstream handles queries intelligently:
   - **Cache Hit**: Data is retrieved directly from the **DuckDB cache**.
   - **Cache Miss**: Queries are routed to ADBC-compliant backend datastores, ensuring reliable data access.
3. Results from cache misses are written back to the cache for future requests.
4. Pre-computed aggregates (e.g., averages, running totals) are served for metrics queries, minimizing processing latency.
5. Queries can retrieve **aggregates by time window** for granular metrics (e.g., sales by hour, user activity by day).

### 3️⃣ Cache Expiry

- **Periodic Evaluation**: Evaluates cached data regularly to maintain optimal resource utilization.
- **Automatic Eviction**: Frees up memory by expiring outdated or unused entries.
- **Window Expiry**: Removes data for expired time windows to conserve resources while retaining recent insights.

## Benefits 🌟

- **🚀 Blazing Fast**: Achieve ultra-low latency with in-memory caching and efficient query routing.
- **⚙️ Scalable**: Handles large-scale data workflows with ease.
- **🔗 Flexible**: Integrates seamlessly with multiple backend systems like Postgres, Redis, and Snowflake.
- **🤖 AI/ML Ready**: Designed to support vectorized data for AI/ML inference pipelines.
- **📈 Real-Time Metrics**: Dynamically calculate and serve statistical metrics (e.g., averages) for monitoring and inference.
- **⌛ Time-Partitioned Insights**: Gain granular control of metrics with support for fixed time windows.
- **⛭ Rust-Powered**: Built using the high-performance, memory-safe Rust programming language, ensuring reliability and speed.

## Comparisons 🆚

**Better together,** Hyprstream is designed to complement Flink and MotherDuck by providing real-time, low-latency answers while they handle batch or complex processing.

Hyprstream is positioned as a bridge between simplicity and power, combining the ease of use of MotherDuck with the real-time capabilities of Flink. It’s ideal for workflows where teams need low-latency insights and streaming features without the full complexity of large-scale event-driven systems.

For example, in a RAG workload, Hyprstream can serve live data or cached insights for quick responses, while Flink processes large-scale data streams for embedding generation, and MotherDuck performs historical trend analysis or offline data prep. This hybrid approach ensures data freshness and responsiveness, with Hyprstream bridging the gap between real-time demands and the medium-to-high latency of Flink and MotherDuck, enabling seamless integration for streaming, batch, and analytics pipelines.

### Hyprstream vs. Apache Flink

| **Feature**               | **Hyprstream**                                                                                   | **Apache Flink**                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Direct Querying**       | ✅ Yes, via Arrow Flight SQL (no external sink needed)                                            | ❌ Must write to external sinks (e.g., Kafka, databases)                                                   |
| **Focus**                 | Real-time aggregation, caching, and ultra-low-latency querying                                    | General-purpose streaming and batch (large-scale event processing)                                         |
| **Integration**           | Microservices-friendly; easy to embed in modern pipelines (Kafka, Spark); minimal configuration   | Integrates with broader ecosystems; requires more setup (clusters, stateful ops)                           |
| **Complexity**            | Low (lightweight deployment, minimal overhead)                                                   | Higher (requires job managers, checkpointing, cluster management)                                          |
| **Latency**               | Ultra-low; queries run in memory with Arrow + DuckDB                                              | Medium-to-high; data typically flows to external sinks before being queried                                |
| **Data Freshness**        | ✅ Near real-time; cached or streaming data ensures up-to-date retrieval                          | ✅ Supports real-time updates but with slightly higher lag due to processing overhead                      |
| **Aggregation Support**   | Basic time windows and metrics; great for real-time dashboards                                    | Advanced, customizable windowing (tumbling, sliding, session windows)                                      |
| **Best For**              | Quick real-time insights, dashboards, streaming metrics, IoT data                                 | Complex, large-scale event-driven architectures, streaming + batch workloads                               |
| **RAG Workloads**         | ✅ Ideal for fast data retrieval and live augmentation                                            | ✅ Suitable for large-scale RAG pipelines; better for batch + stream combination workflows                  |
| **Continuous Bias Training**| ✅ Excellent for streaming bias detection and incremental updates                                | ✅ Good for both real-time bias updates and large-scale batch retraining                                    |
| **ML Batch Workloads**    | Feeds small, real-time data slices to external ML frameworks; no built-in batch processing        | Built-in batch APIs (Table/DataSet) for large-scale ML preprocessing                                       |
| **ML Real-Time Inference**| ✅ Ideal for on-the-fly scoring (microservices)                                                   | ✅ Can handle large-scale streaming inference with higher overhead                                          |
| **ML Tools Integration**  | Arrow-based data export to Python/R or other ML frameworks; easy microservice integration         | Integrates with Flink ML libs and external frameworks for both batch and streaming ML                      |

---

### Hyprstream vs. MotherDuck

| **Feature**               | **Hyprstream**                                                                                      | **MotherDuck**                                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Direct Querying**       | ✅ Yes, via Arrow Flight SQL (real-time)                                                            | ✅ In-process DuckDB + serverless cloud queries                                                      |
| **Focus**                 | Real-time aggregation, caching, and sub-second querying                                            | Hybrid local-cloud analytics (batch and interactive)                                                 |
| **Integration**           | Fits easily with streaming pipelines (Flink, Kafka) and microservices                              | Integrates with BI/data tools and can offload big data to the cloud                                  |
| **Complexity**            | Low (lightweight microservice or embedded library)                                                 | Low/Medium (local DuckDB + optional cloud scaling)                                                   |
| **Latency**               | Ultra-low (in-memory operations, optimized for streaming data)                                     | Medium; suited for interactive queries but not optimized for sub-second real-time use                |
| **Data Freshness**        | ✅ Real-time; supports live streaming and cached data                                              | ❌ Batch or semi-static; relies on periodic updates for freshness                                    |
| **Aggregation Support**   | Basic time windows, rolling metrics, real-time updates                                             | Full SQL with DuckDB, scalable to larger datasets when using cloud resources                         |
| **Best For**              | Real-time dashboards, IoT metrics, live data monitoring                                            | Ad-hoc SQL analytics, BI reporting, hybrid on-prem/cloud workloads                                   |
| **RAG Workloads**         | ✅ Perfect for live data augmentation in real-time RAG systems                                      | ❌ Better suited for static/batch RAG use cases                                                     |
| **Continuous Bias Training**| ✅ Strong in streaming bias metrics and preprocessing for adaptive models                        | ❌ Limited to batch bias correction workflows                                                        |
| **ML Batch Workloads**    | Supplies real-time feature data for external ML; does not provide large-scale batch engine         | Well-suited for batch data prep; can push big datasets to cloud for ML training                      |
| **ML Real-Time Inference**| ✅ Perfect for quick, microservice-based model scoring                                             | ❌ More aligned with batch or interactive queries; limited continuous streaming support               |
| **ML Tools Integration**  | Arrow-native data; easily exported to Python/R or microservices for ML pipelines                   | Relies on DuckDB for data prep and merges well with local or cloud ML environments                   |

## Configuration 🔧

Hyprstream uses TOML configuration files and environment variables for configuration. The configuration is loaded in the following order:

1. Default configuration (embedded in binary from `config/default.toml`)
2. System-wide configuration (`/etc/hyprstream/config.toml` - optional)
3. User-specified configuration file (via `--config` CLI option - optional)
4. Environment variables (prefixed with `HYPRSTREAM_`)
5. Command-line arguments (highest precedence)

### Environment Variables

The following environment variables are supported:

```bash
# Server configuration
HYPRSTREAM_SERVER_HOST=0.0.0.0
HYPRSTREAM_SERVER_PORT=8080

# Cache configuration
HYPRSTREAM_CACHE_DURATION=7200

# ADBC configuration
HYPRSTREAM_ADBC_DRIVER_PATH=/usr/local/lib/libadbc_driver_postgresql.so
HYPRSTREAM_ADBC_URL=postgresql://db.example.com:5432
HYPRSTREAM_ADBC_USERNAME=app_user
HYPRSTREAM_ADBC_DATABASE=metrics
```

### Command Line Arguments

All configuration options can also be set via command line arguments:

```bash
hyprstream \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-duration 7200 \
  --driver-path /usr/local/lib/libadbc_driver_postgresql.so \
  --db-url postgresql://db.example.com:5432 \
  --db-user app_user \
  --db-name metrics \
  --config /path/to/custom/config.toml
```

### Configuration File Example

Here's an example of the TOML configuration file format:

```toml
[server]
host = "127.0.0.1"
port = 50051

[cache]
duration_secs = 3600  # 1 hour

[adbc]
driver_path = "/usr/local/lib/libadbc_driver_postgresql.so"
url = "postgresql://localhost:5432"
username = "postgres"
password = ""
database = "metrics"

# Optional: Database-specific configurations
[adbc.options]
application_name = "hyprstream"
connect_timeout = "10"

# Optional: Connection pool settings
[adbc.pool]
max_connections = 10
min_connections = 1
acquire_timeout_secs = 30
```

Note: Command line arguments and environment variables take precedence over configuration files. The `--config` option allows you to specify a custom configuration file path.

## Example Usage 💡

To get started, check out our **[Python Client Example](examples/client/python)**. This example demonstrates how to:

- Ingest data into Hyprstream using Arrow Flight.
- Query data with Arrow Flight SQL.
- Interact with the DuckDB cache and underlying ADBC datastores.
- Query pre-computed aggregates for real-time metrics.
- Retrieve aggregate metrics for specific **time windows** (e.g., last hour, past 7 days).

## Environment Variables 🔧

Hyprstream uses environment variables for configuring the ADBC backend connection. These must be set before starting the server:

### Required Variables

- `ADBC_DRIVER`: Path to the ADBC driver shared library (e.g., `/usr/local/lib/libadbc_postgres.so`)
- `ADBC_URL`: Database connection URL (e.g., `postgresql://localhost:5432`)

### Optional Variables

- `ADBC_USERNAME`: Database username (defaults to empty string)
- `ADBC_PASSWORD`: Database password (defaults to empty string)
- `ADBC_DATABASE`: Database name (defaults to empty string)

### Example Configuration

```bash
# PostgreSQL Example
export ADBC_DRIVER=/usr/local/lib/libadbc_postgres.so
export ADBC_URL=postgresql://localhost:5432
export ADBC_USERNAME=myuser
export ADBC_PASSWORD=mypassword
export ADBC_DATABASE=metrics

# SQLite Example
export ADBC_DRIVER=/usr/local/lib/libadbc_sqlite.so
export ADBC_URL=file:metrics.db

# Snowflake Example
export ADBC_DRIVER=/usr/local/lib/libadbc_snowflake.so
export ADBC_URL=https://myorg.snowflakecomputing.com
export ADBC_USERNAME=myuser
export ADBC_PASSWORD=mypassword
export ADBC_DATABASE=MYDB
```

The ADBC backend will automatically use these environment variables to establish a connection to your chosen database. Make sure the appropriate ADBC driver is installed and accessible at the specified path.

---
For inquiries or support, contact us at **[support@hyprstream.com](mailto:support@hyprstream.com)** or visit our GitHub repository to contribute! 🌐