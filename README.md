# Hyprstream: Real-time Aggregation Windows and High-Performance Cache for Apache Arrow Flight SQL 🚀

Hyprstream is a next-generation application for real-time data ingestion, windowed aggregation, caching, and serving. Built on Apache Arrow Flight and DuckDB, and developed in Rust, Hyprstream dynamically calculates metrics like running sums, counts, and averages, enabling blazing-fast data workflows, intelligent caching, and seamless integration with ADBC-compliant datastores. Its real-time aggregation capabilities empower AI/ML pipelines and analytics with instant insights. 💾✨

---

## Key Features 🎯

### 🔄 Data Ingestion via Apache Arrow Flight

- **Streamlined Ingestion**: Ingests data efficiently using **Arrow Flight**, an advanced columnar data transport protocol.
- **Real-Time Streaming**: Supports real-time metrics, datasets, and vectorized data, making it perfect for analytics and AI/ML workflows.
- **Seamless Integration**: Works effortlessly with data producers for high-throughput ingestion.
- **Write-Through to ADBC Datastores**: Ensures eventual data consistency by writing cache-miss results to backend datastores such as Postgres, Redis, and Snowflake via ADBC.

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

---

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

---

## Benefits 🌟

- **🚀 Blazing Fast**: Achieve ultra-low latency with in-memory caching and efficient query routing.
- **⚙️ Scalable**: Handles large-scale data workflows with ease.
- **🔗 Flexible**: Integrates seamlessly with multiple backend systems like Postgres, Redis, and Snowflake.
- **🤖 AI/ML Ready**: Designed to support vectorized data for AI/ML inference pipelines.
- **📈 Real-Time Metrics**: Dynamically calculate and serve statistical metrics (e.g., averages) for monitoring and inference.
- **⌛ Time-Partitioned Insights**: Gain granular control of metrics with support for fixed time windows.
- **⛭ Rust-Powered**: Built using the high-performance, memory-safe Rust programming language, ensuring reliability and speed.

---

## Example Usage 💡

To get started, check out our **[Python Client Example](examples/client/python)**. This example demonstrates how to:

- Ingest data into Hyprstream using Arrow Flight.
- Query data with Arrow Flight SQL.
- Interact with the DuckDB cache and underlying ADBC datastores.
- Query pre-computed aggregates for real-time metrics.
- Retrieve aggregate metrics for specific **time windows** (e.g., last hour, past 7 days).

---

For inquiries or support, contact us at **[support@hyprstream.com](mailto:support@hyprstream.com)** or visit our GitHub repository to contribute! 🌐