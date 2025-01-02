import pyarrow.flight as flight
import pyarrow as pa
import time

def main():
    print("Connecting to Flight server")
    client = flight.FlightClient("grpc://localhost:50051")

    # Create a time window query
    end_time = int(time.time() * 1e9)  # current time in nanoseconds
    start_time = end_time - (60 * 1e9)  # 60 seconds earlier
    print(f"\nQuerying data for time window: {start_time} to {end_time}")
    
    # Execute SQL query - must exactly match parse_sql_timestamps format
    query = f"""select * from metrics where timestamp >= {start_time} and timestamp <= {end_time}"""
    
    try:
        # Execute the query and get results
        info = flight.Action("ExecuteSql", query.encode("utf-8"))
        results = client.do_action(info)
        #reader = client.do_get(info.endpoints[0].ticket)
        
        # Process results
        for batch in results:
            df = batch.to_pandas()
            print("\nQuery results:")
            print(df)
            
            # The schema matches storage.rs:
            # - metric_id (string)
            # - timestamp (timestamp[ns]) 
            # - valueRunningWindowSum (float64)
            # - valueRunningWindowAvg (float64)
            # - valueRunningWindowCount (int64)
            
    except flight.FlightError as e:
        print(f"Flight error: {e}")

if __name__ == "__main__":
    main()