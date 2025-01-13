use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

// Import test utilities
use crate::test_utils::{ThreadSafetyTester, async_test};

#[tokio::test]
async fn test_concurrent_counter_access() {
    let tester = ThreadSafetyTester::new();
    let threads = 100;
    let count = tester.concurrent_access(threads).await;
    assert_eq!(count, threads, "Counter should match number of threads");
}

#[tokio::test]
async fn test_mutex_data_race_prevention() {
    let shared_data = Arc::new(Mutex::new(vec![]));
    let mut handles = vec![];

    for i in 0..10 {
        let data = shared_data.clone();
        handles.push(tokio::spawn(async move {
            let mut guard = data.lock().unwrap();
            guard.push(i);
            sleep(Duration::from_millis(10)).await;
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let final_data = shared_data.lock().unwrap();
    assert_eq!(final_data.len(), 10, "All updates should be recorded");
}

#[tokio::test]
async fn test_rwlock_synchronization() {
    let shared_data = Arc::new(RwLock::new(0));
    let mut read_handles = vec![];
    let mut write_handles = vec![];

    // Spawn writer threads
    for i in 0..5 {
        let data = shared_data.clone();
        write_handles.push(tokio::spawn(async move {
            let mut guard = data.write().await;
            *guard += i;
            sleep(Duration::from_millis(10)).await;
        }));
    }

    // Spawn reader threads
    for _ in 0..10 {
        let data = shared_data.clone();
        read_handles.push(tokio::spawn(async move {
            let guard = data.read().await;
            assert!(*guard >= 0, "Value should never be negative");
        }));
    }

    // Wait for all writers
    for handle in write_handles {
        handle.await.unwrap();
    }

    // Wait for all readers
    for handle in read_handles {
        handle.await.unwrap();
    }

    let final_value = *shared_data.read().await;
    assert_eq!(final_value, 10, "Final value should be sum of all writes");
}

#[tokio::test]
async fn test_thread_pool_behavior() {
    let pool_size = 4;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(pool_size)
        .build()
        .unwrap();

    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    // Submit more tasks than threads
    for _ in 0..pool_size * 2 {
        let counter = counter.clone();
        handles.push(runtime.spawn(async move {
            let mut guard = counter.lock().unwrap();
            *guard += 1;
            sleep(Duration::from_millis(100)).await;
        }));
    }

    // Wait for all tasks
    for handle in handles {
        runtime.block_on(handle).unwrap();
    }

    let final_count = *counter.lock().unwrap();
    assert_eq!(final_count, pool_size * 2, "All tasks should complete");
}

#[tokio::test]
async fn test_deadlock_prevention() {
    let resource1 = Arc::new(Mutex::new(0));
    let resource2 = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    // Create potential deadlock scenario
    for i in 0..2 {
        let r1 = resource1.clone();
        let r2 = resource2.clone();
        handles.push(tokio::spawn(async move {
            if i % 2 == 0 {
                let _guard1 = r1.lock().unwrap();
                sleep(Duration::from_millis(10)).await;
                let _guard2 = r2.lock().unwrap();
            } else {
                let _guard2 = r2.lock().unwrap();
                sleep(Duration::from_millis(10)).await;
                let _guard1 = r1.lock().unwrap();
            }
        }));
    }

    // Use timeout to detect deadlocks
    let result = tokio::time::timeout(Duration::from_secs(1), async {
        for handle in handles {
            handle.await.unwrap();
        }
    }).await;

    assert!(result.is_ok(), "Deadlock prevention should allow completion within timeout");
}
