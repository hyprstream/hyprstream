use hyprstream_core::aggregation::TimeWindow;
use std::time::Duration;

#[test]
fn test_window_bounds_none() {
    let window = TimeWindow::None;
    let timestamp = 1000;
    let (start, end) = window.window_bounds(timestamp);
    assert_eq!(start, i64::MIN);
    assert_eq!(end, i64::MAX);
}

#[test]
fn test_window_bounds_fixed() {
    let window = TimeWindow::Fixed(Duration::from_secs(60)); // 1 minute window
    let timestamp = 100; // Should round down to nearest minute
    let (start, end) = window.window_bounds(timestamp);
    assert_eq!(start, 60); // Rounds down to nearest 60
    assert_eq!(end, 120); // Next window boundary
}

#[test]
fn test_window_bounds_sliding() {
    let window = TimeWindow::Sliding {
        window: Duration::from_secs(300), // 5 minute window
        slide: Duration::from_secs(60),   // 1 minute slide
    };
    let timestamp = 100;
    let (start, end) = window.window_bounds(timestamp);
    assert_eq!(start, 60);  // Rounds down to nearest slide
    assert_eq!(end, 360);   // Window size (300) added to start
}

#[test]
fn test_window_sql_generation() {
    // Test None window
    let window = TimeWindow::None;
    assert!(window.to_sql().is_none());

    // Test Fixed window
    let window = TimeWindow::Fixed(Duration::from_secs(60));
    let sql = window.to_sql().unwrap();
    assert!(sql.contains("timestamp / 60"));
    assert!(sql.contains("window_start"));
    assert!(sql.contains("window_end"));

    // Test Sliding window
    let window = TimeWindow::Sliding {
        window: Duration::from_secs(300),
        slide: Duration::from_secs(60),
    };
    let sql = window.to_sql().unwrap();
    assert!(sql.contains("timestamp / 60"));
    assert!(sql.contains("window_start"));
    assert!(sql.contains("window_end"));
    assert!(sql.contains("300")); // Window size
} 