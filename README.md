# Hyprstream: Real-time Aggregation and High-Performance Cache for Apache Arrow Flight SQL 🚀

Hyprstream is a next-generation application for real-time data ingestion, aggregation, caching, and serving. Built on Apache Arrow Flight and DuckDB, Hyprstream dynamically calculates metrics like running sums, counts, and averages, enabling blazing-fast data workflows, intelligent caching, and seamless integration with ADBC-compliant datastores. Its real-time aggregation capabilities empower AI/ML pipelines and analytics with instant insights. 💾✨

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

---

## Application Workflow 🔧

### 1️⃣ Data Ingestion

1. Data producers send data to Hyprstream via the **Arrow Flight ingestion API**.
2. The data is processed and stored in the in-memory **DuckDB cache**.
3. Aggregates for sums, counts, and averages are dynamically updated as new data arrives.

### 2️⃣ Query Processing

1. Consumers send queries via **Arrow Flight SQL**.
2. Hyprstream handles queries intelligently:
   - **Cache Hit**: Data is retrieved directly from the **DuckDB cache**.
   - **Cache Miss**: Queries are routed to ADBC-compliant backend datastores, ensuring reliable data access.
3. Results from cache misses are written back to the cache for future requests.
4. Pre-computed aggregates (e.g., averages, running totals) are served for metrics queries, minimizing processing latency.

### 3️⃣ Cache Expiry

- **Periodic Evaluation**: Evaluates cached data regularly to maintain optimal resource utilization.
- **Automatic Eviction**: Frees up memory by expiring outdated or unused entries.

---

## Benefits 🌟

- **🚀 Blazing Fast**: Achieve ultra-low latency with in-memory caching and efficient query routing.
- **⚙️ Scalable**: Handles large-scale data workflows with ease.
- **🔗 Flexible**: Integrates seamlessly with multiple backend systems like Postgres, Redis, and Snowflake.
- **🤖 AI/ML Ready**: Designed to support vectorized data for AI/ML inference pipelines.
- **📈 Real-Time Metrics**: Dynamically calculate and serve statistical metrics (e.g., averages) for monitoring and inference.

---

## Example Usage 💡

To get started, check out our **[Python Client Example](examples/client/python)**. This example demonstrates how to:

- Ingest data into Hyprstream using Arrow Flight.
- Query data with Arrow Flight SQL.
- Interact with the DuckDB cache and underlying ADBC datastores.
- Query pre-computed aggregates for real-time metrics.

---

For inquiries or support, contact us at **[support@hyprstream.com](mailto:support@hyprstream.com)** or visit our GitHub repository to contribute! 🌐