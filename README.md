# Hyprstream: Real-time Aggregation Windows and High-Performance Cache for Apache Arrow Flight SQL 🚀

[![Rust](https://github.com/hyprstream/hyprstream/actions/workflows/rust.yml/badge.svg)](https://github.com/hyprstream/hyprstream/actions/workflows/rust.yml)

📄 **[Read our DRAFT Technical Paper: Hyprstream: A Unified Architecture for Multimodal Data Processing and Real-Time Foundational Model Inference](https://github.com/hyprstream/hyprstream/blob/main/HYPRSTREAM_PAPER_DRAFT.md)**

⚠️ **PRE-RELEASE: This is a work in progress alpha and is not yet ready for production and is in early stages of development** ⚠️ ️

🌟 Hyprstream is a next-generation application for real-time data ingestion, windowed aggregation, caching, and serving. Built on Apache Arrow Flight and DuckDB, and developed in Rust, Hyprstream dynamically calculates metrics like running sums, counts, and averages, enabling blazing-fast data workflows, intelligent caching, and seamless integration with ADBC-compliant datastores. Its real-time aggregation capabilities empower AI/ML pipelines and analytics with instant insights. 💾✨

🚧 **This product is in preview during rapid early development**. While we're laying the groundwork for advertised capabilities, there are known bugs 🐛, partially implemented features 🔨, and frequent updates ahead 🔄. Your feedback and collaboration will be invaluable in shaping the project's direction 🌱.

## ✨ Key Features

### 📥 Data Ingestion via Apache Arrow Flight

- 🔄 **Streamlined Ingestion**: Ingests data efficiently using **Arrow Flight**, an advanced columnar data transport protocol
- ⚡ **Real-Time Streaming**: Supports real-time metrics, datasets, and vectorized data for analytics and AI/ML workflows
- 💾 **Write-Through to ADBC**: Ensures data consistency with immediate caching and write-through to backend datastores

### 🧠 Intelligent Read Caching with DuckDB

- ⚡ **In-Memory Performance**: Uses **DuckDB** for lightning-fast caching of frequently accessed data
- 🎯 **Optimized Querying**: Stores query results and intermediate computations for analytics workloads
- 🔄 **Automatic Management**: Handles caching transparently with configurable expiry policies

### 📊 Real-Time Aggregation

- 📈 **Dynamic Metrics**: Maintains running sums, counts, and averages for real-time insights
- ⏱️ **Time Window Partitioning**: Supports fixed time windows (e.g., 5m, 30m, hourly, daily) for granular analysis
- 🎯 **Lightweight State**: Maintains only aggregate states for efficient memory usage

### 🌐 Data Serving with Arrow Flight SQL

- ⚡ **High-Performance Queries**: Serves cached data via Arrow Flight SQL for minimal latency
- 🔢 **Vectorized Data**: Optimized for AI/ML pipelines and analytical queries
- 🔌 **Seamless Integration**: Connects with analytics and visualization tools

## 🌟 Benefits

- ⚡ **Low Latency**: Millisecond-level query responses for cached data
- 📈 **Scalable**: Handles large-scale data workflows with ease
- 🔗 **Flexible**: Integrates with Postgres, Redis, Snowflake, and other ADBC datastores
- 🤖 **AI/ML Ready**: Optimized for vectorized data and inference pipelines
- 📊 **Real-Time Metrics**: Dynamic calculation of statistical metrics
- ⏱️ **Time Windows**: Granular control of metrics with configurable windows
- 🦀 **Rust-Powered**: High-performance, memory-safe implementation

## 🔜 Coming Soon

Hyprstream is actively developing several exciting features:

### 🧠 Real-Time Model Integration
- 📦 Direct storage of foundational models in Arrow format
- 🚀 Zero-copy GPU access for model weights
- 🔄 Layer-specific updates and fine-tuning

### 🔮 Advanced Processing
- 🔄 Multimodal data fusion with real-time embedding generation
- ⚡ CUDA-optimized operations with custom kernels
- 📊 Advanced time-series window operations
- 🎥 Neural Radiance Fields (NERF) integration for video processing

### 🚀 Performance & Scale
- 📦 Multi-tiered storage system with intelligent caching
- 🌐 Distributed training and gradient accumulation
- ⚡ GPU-accelerated query execution
- 🔄 Predictive layer prefetching

### 🔒 Security & Privacy
- 🔐 Encrypted model weight storage
- 🛡️ Privacy-preserving training with differential privacy
- 📝 Comprehensive audit logging
- 🔑 Fine-grained access control

For detailed technical information about these upcoming features, please refer to our [technical paper](HYPRSTREAM_PAPER_DRAFT.md).

### 📊 Ecosystem Integration

Hyprstream is designed to work seamlessly with existing data infrastructure:

#### 🔗 Storage & Analytics
- Works with any ADBC-compliant database (PostgreSQL, Snowflake, etc.) as a backend store
- Uses DuckDB for high-performance caching and analytics
- Integrates with Arrow ecosystem tools for data processing and analysis

#### 🔄 Real-time Processing
- Complements stream processing systems by providing fast caching layer
- Can serve as a real-time metrics store for monitoring solutions
- Enables quick access to recent data while maintaining historical records

#### 🤖 AI/ML Pipeline Integration (Coming Soon)
- Will provide zero-copy access to model weights and embeddings
- Designed to work alongside vector databases and ML serving platforms
- Future support for real-time model updates and fine-tuning

#### 🛠️ Developer Tools
- Native Arrow Flight SQL support for seamless client integration
- Compatible with popular data science tools and frameworks
- Language-agnostic API for broad ecosystem compatibility

Hyprstream focuses on being a great citizen in the modern data stack, enhancing rather than replacing existing tools.

## 🚀 Getting Started

1. 📥 Install Hyprstream:
   ```bash
   cargo install hyprstream
   ```

2. 🏃 Start the server with default configuration:
   ```bash
   hyprstream
   ```

3. 🔌 Use with PostgreSQL backend (requires PostgreSQL ADBC driver):
   ```bash
   # Set backend-specific credentials securely via environment variables
   export HYPRSTREAM_ENGINE_USERNAME=postgres
   export HYPRSTREAM_ENGINE_PASSWORD=secret

   # Start Hyprstream with connection details (but without credentials)
   hyprstream \
     --engine adbc \
     --engine-connection "postgresql://localhost:5432/metrics?pool_max=10&pool_min=1&connect_timeout=30" \
     --engine-options driver_path=/usr/local/lib/libadbc_driver_postgresql.so \
     --enable-cache \
     --cache-engine duckdb \
     --cache-connection ":memory:"
   ```

For configuration options and detailed documentation, run:
```bash
hyprstream --help
```

Or visit our [📚 API Documentation](https://docs.rs/hyprstream) for comprehensive guides and examples.

## 💡 Example Usage

### 🚀 Quick Start with ADBC

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

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---
For inquiries or support, contact us at [📧 support@hyprstream.com](mailto:support@hyprstream.com) or visit our GitHub repository to contribute! 🌐