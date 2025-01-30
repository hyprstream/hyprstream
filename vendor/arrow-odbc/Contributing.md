# Contributions

Whether they be in code, interesting feature suggestions, design critique or bug reports, all contributions are welcome. Please start an issue, before investing a lot of work. This helps avoid situations there I would feel the need to reject a large body of work, and a lot of your time has been wasted. `odbc-arrow` is a pet project and a work of love, which implies that I maintain it in my spare time. Please understand that I may not always react immediately. If you contribute code to fix a Bug, please also contribute the test to fix it. Happy contributing.

## Local build and test setup

Running local tests currently requires:

* Docker and Docker compose.
* An ODBC driver manager
* A driver for Microsoft SQL Server
* Rust toolchain (cargo)

You can install these requirements from here:

* Install Rust compiler and Cargo. Follow the instructions on [this site](https://www.rust-lang.org/en-US/install.html).
* [Microsoft ODBC Driver 17 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15).
* An ODBC Driver manager if you are not on windows: <http://www.unixodbc.org/>

With docker installed run:

```shell
docker-compose up
```

This starts the Microsoft SQL Server used for testing.

We now can execute the tests in Rust typical fashion using:

```shell
cargo test
```
