
import pandas as pd


data = {
    'instructions': 
    [
        {'test': 6, 'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 1, 'PrePress2Buffer': 1, 'PostPress12Buffer': 1, 'throughput': 28.114285714285717, 'wip': 10.779661016949152}, 
        {'test': 7, 'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 1, 'PrePress2Buffer': 3, 'PostPress12Buffer': 1, 'throughput': 30.17142857142857, 'wip': 12.836158192090396}, 
        {'test': 8, 'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 1, 'PrePress1Buffer': 2, 'PrePress2Buffer': 1, 'PostPress12Buffer': 1, 'throughput': 29.485714285714288, 'wip': 11.768361581920905}
    ]
    }

data2 = {
    'instructions': 
    [
        {'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 3, 'PrePress1Buffer': 3, 'PrePress2Buffer': 2, 'PostPress12Buffer': 3, 'throughput': 27.24404761904762, 'wip': 16.16483134920635}, 
        {'PostLoadingBuffer': 2, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 2, 'PrePress1Buffer': 3, 'PrePress2Buffer': 2, 'PostPress12Buffer': 2, 'throughput': 27.23809523809524, 'wip': 16.01919642857143}, 
        {'PostLoadingBuffer': 1, 'PostConveyorBuffer': 1, 'PostWashingBuffer': 2, 'PrePress1Buffer': 2, 'PrePress2Buffer': 2, 'PostPress12Buffer': 2, 'throughput': 27.154761904761905, 'wip': 13.998561507936508}
    ]
    }


def json_to_csv(json_data, filename="suggested_improvements.csv"):
    try:
        # 1. Dynamically find the key that holds the list (e.g., 'instructions')
        # This takes the first key it finds in the dictionary
        root_key = list(json_data.keys())[0]
        records = json_data[root_key]

        # 2. Convert to DataFrame
        # Pandas automatically extracts column names from the dictionary keys
        df = pd.DataFrame(records)

        # 3. Save to the main folder
        df.to_csv(filename, index=False)
        
        print(f"✅ Successfully saved {len(df)} datapoints to '{filename}'")
        print(f"Columns identified: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error converting JSON to CSV: {e}")

# Usage
json_to_csv(data2)