#[cfg(test)]
mod tests {
    use assert_cmd::Command;
    use predicates::prelude::*;

    #[test]
    fn test_help() {
        let mut cmd = Command::cargo_bin("hyprstream").unwrap();
        cmd.arg("--help");
        cmd.assert()
            .success()
            .stdout(predicate::str::contains("Usage:"))
            .stdout(predicate::str::contains("Commands:"));
    }

    #[test]
    fn test_server_args() {
        let mut cmd = Command::cargo_bin("hyprstream").unwrap();
        cmd.args(["server", "--listen=127.0.0.1:8080"]);
        cmd.assert().success();
    }

    #[test] 
    fn test_models_list() {
        let mut cmd = Command::cargo_bin("hyprstream").unwrap();
        cmd.args(["models", "list"]);
        cmd.assert().success();
    }

    #[test]
    fn test_models_tag() {
        let mut cmd = Command::cargo_bin("hyprstream").unwrap();
        cmd.args(["models", "tag", "model1", "v1"]);
        cmd.assert().success();
    }

    #[test]
    fn test_chat() {
        let mut cmd = Command::cargo_bin("hyprstream").unwrap();
        cmd.args(["chat", "model1:v1"]);
        cmd.assert().success();
    }

    #[test]
    fn test_sql() {
        let mut cmd = Command::cargo_bin("hyprstream").unwrap();
        cmd.args(["sql", "--host=127.0.0.1:8080"]);
        cmd.assert().success();
    }
}
use assert_cmd::Command;

#[test]
fn test_help_command() {
    let mut cmd = Command::cargo_bin("hyprstream").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("Usage:"));
}

#[test]
fn test_server_debug() {
    let mut cmd = Command::cargo_bin("hyprstream").unwrap();
    cmd.args(["server", "--debug"])
        .assert()
        .success();
}

#[test]
fn test_models_list() {
    let mut cmd = Command::cargo_bin("hyprstream").unwrap();
    cmd.args(["models", "list"])
        .assert()
        .success();
}

#[test]
fn test_chat_with_model() {
    let mut cmd = Command::cargo_bin("hyprstream").unwrap();
    cmd.args(["chat", "--model-tag", "gpt3:latest"])
        .assert()
        .success();
}

#[test]
fn test_sql_with_host() {
    let mut cmd = Command::cargo_bin("hyprstream").unwrap();
    cmd.args(["sql", "--host", "localhost:50051"])
        .assert()
        .success();
}
