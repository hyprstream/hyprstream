This page describes the product and the parameters for acceptance criteria for
the hyprstream project.

The CLI options below are non-exclusive.

1. provide a cli with an experience as described:
1a. hyprstream server [--listen=<host:port>|-d] # -d provides detach
1b. hyprstream models [list|import|delete|inspect] <model:tag>
1b1. hyprstream models tag <model_id> <model:tag>
1c. hyprstream chat <model:tag>
1d. hyprstream sql [--host=<host:port>] # defaults to localhost
2. a chat interface via `hyprstream chat <model:tag>`
2a. chats should be able to access and query inflight data by applying layers and making transforms between tables and LLM inputs, data from the vector database backend (cached, duckdb, or adbc) and if requested or necessary to serve the query, should use an optimized time-series window selection.
3. implement and maintain tests
4. run `cargo check` to verify your code. do not attmept to run tests unless `cargo check` is successful. the project is complete when the HYPRSTREAM_PAPER_DRAFT.md has tests respresenting and implemetning the described systems and software, the code successfully completes a `cargo check` without errors, and all test pass. The software should have no warnings unless approved by a human in the loop or explicitly listed below:
 (warnings excepted: NONE)
