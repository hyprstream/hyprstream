use hyprstream_core::aggregation::{GroupBy, TimeWindow};
use std::time::Duration;

#[test]
fn test_group_by_no_window() {
    let group_by = GroupBy {
        columns: vec!["metric".to_string(), "host".to_string()],
        time_column: None,
    };
    
    let sql = group_by.to_sql();
    assert!(sql.contains("GROUP BY metric, host"));
    assert!(!sql.contains("window_start"));
}

#[test]
fn test_group_by_fixed_window() {
    let group_by = GroupBy {
        columns: vec!["metric".to_string()],
        time_column: Some("timestamp".to_string()),
    };
    
    let window = TimeWindow::Fixed(Duration::from_secs(60));
    let sql = group_by.to_sql_with_window(&window);
    
    assert!(sql.contains("GROUP BY metric"));
    assert!(sql.contains("window_start"));
    assert!(sql.contains("timestamp / 60"));
}

#[test]
fn test_group_by_sliding_window() {
    let group_by = GroupBy {
        columns: vec!["metric".to_string(), "host".to_string()],
        time_column: Some("timestamp".to_string()),
    };
    
    let window = TimeWindow::Sliding {
        window: Duration::from_secs(300),
        slide: Duration::from_secs(60),
    };
    let sql = group_by.to_sql_with_window(&window);
    
    assert!(sql.contains("GROUP BY metric, host"));
    assert!(sql.contains("window_start"));
    assert!(sql.contains("timestamp / 60"));
    assert!(sql.contains("300")); // Window size
}

#[test]
fn test_group_by_empty() {
    let group_by = GroupBy {
        columns: vec![],
        time_column: None,
    };
    
    let sql = group_by.to_sql();
    assert!(sql.is_empty());
}

#[test]
fn test_group_by_time_only() {
    let group_by = GroupBy {
        columns: vec![],
        time_column: Some("timestamp".to_string()),
    };
    
    let window = TimeWindow::Fixed(Duration::from_secs(60));
    let sql = group_by.to_sql_with_window(&window);
    
    assert!(sql.contains("GROUP BY"));
    assert!(sql.contains("window_start"));
    assert!(!sql.contains("metric"));
} 