import os
import pandas as pd

def main():
    print("=" * 50)
    print("CSV TRUNCATOR")
    print("=" * 50)
    
    # 1. Get file path
    print("\nEnter the path to your CSV file:")
    file_path = input("> ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    # 2. Get number of rows
    print("\nHow many rows do you want to keep?")
    try:
        num_rows = int(input("> "))
        if num_rows <= 0:
            print("Error: Please enter a positive number.")
            return
    except ValueError:
        print("Error: Invalid number.")
        return

    print("\nProcessing...")
    
    try:
        # Read only the specified number of rows (+1 for header to handle correctly, 
        # but pandas nrows applies to data rows usually)
        # Using iterator to handle huge files without memory issues, but "keep it simple" -> just read nrows
        df = pd.read_csv(file_path, nrows=num_rows)
        
        # Generate output filename
        dir_name, file_name = os.path.split(file_path)
        name, ext = os.path.splitext(file_name)
        output_file = os.path.join(dir_name, f"{name}_{num_rows}rows{ext}")
        
        # Save to new file
        df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 50)
        print("SUCCESS!")
        print("=" * 50)
        print(f"Created: {output_file}")
        print(f"Rows:    {len(df)}")
        
    except Exception as e:
        print(f"\nError processing file: {e}")

if __name__ == "__main__":
    main()
