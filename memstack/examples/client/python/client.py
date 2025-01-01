import pyarrow.flight as flight
import pyarrow as pa
import time

def main():
    print("Connecting to Flight server")
    # Connect to Flight server
    client = flight.FlightClient("grpc://localhost:50051")
    # Write some test data to verify connection
    print("Writing test data...")
    test_data = {
        'metric_id': ['test_metric1', 'test_metric2'],
        'timestamp': [int(time.time() * 1e9)] * 2,
        'valueRunningWindowSum': [100.0, 200.0],
        'valueRunningWindowAvg': [50.0, 100.0], 
        'valueRunningWindowCount': [2, 2]
    }
    
    # Convert to Arrow table
    table = pa.Table.from_pydict(test_data)
    
    # Create a FlightDescriptor for the put operation
    descriptor = flight.FlightDescriptor.for_path(b"test_data")
    
    # Write data using do_put with both descriptor and schema
    writer, _ = client.do_put(descriptor, table.schema)
    writer.write_table(table)
    writer.close()
    print("Test data written successfully")

    # Create a time window query
    end_time = int(time.time() * 1e9)  # current time in nanoseconds
    start_time = end_time - (60 * 1e9)  # 60 seconds earlier
    print(f"Requesting data for time window: {start_time} to {end_time}")
    
    # Format the ticket with start and end times separated by a comma
    ticket = flight.Ticket(f"{start_time},{end_time}".encode())

    try:
        reader = client.do_get(ticket)
        for batch in reader:
            df = batch.to_pandas()
            # The schema matches storage.rs:
            # - metric_id (string)
            # - timestamp (timestamp[ns]) 
            # - valueRunningWindowSum (float64)
            # - valueRunningWindowAvg (float64)
            # - valueRunningWindowCount (int64)
            print(df)
    except flight.FlightInternalError as e:
        print(f"Flight error: {e}")
        # Maybe try with a different timestamp or check server status

if __name__ == "__main__":
    main()