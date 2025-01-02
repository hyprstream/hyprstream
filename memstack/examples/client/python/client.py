import pyarrow as pa
import adbc_driver_flightsql
import adbc_driver_flightsql.dbapi
import time

def main():
    print("Connecting to Flight SQL server")
    
    # Connect using ADBC Flight SQL driver
    conn = adbc_driver_flightsql.dbapi.connect("grpc://localhost:50051")
    cursor = conn.cursor()

    try:
        # Create a time window query
        end_time = int(time.time() * 1e9)  # current time in nanoseconds
        start_time = end_time - (60 * 1e9)  # 60 seconds earlier
        print(f"\nQuerying data for time window: {start_time} to {end_time}")
        
        # Execute SQL query using ADBC
        query = f"""
        select * from metrics 
        where timestamp >= {start_time} 
        and timestamp <= {end_time}
        """
        
        # Execute query and fetch results
        cursor.execute(query)
        results = cursor.fetch_arrow_table()
        
        # Convert to pandas and display
        if results.num_rows > 0:
            df = results.to_pandas()
            print("\nQuery results:")
            print(df)
        else:
            print("\nNo results found")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()