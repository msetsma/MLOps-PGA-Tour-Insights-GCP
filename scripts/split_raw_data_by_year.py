import csv
import os

def split_csv_by_season(input_file, output_dir):
    """Split a CSV file into separate files based on the 'season' column."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        # A dictionary to hold data for each season
        data_by_season = {}

        for row in reader:
            season = row.get('season')
            if not season:
                print(f"Skipping row with missing 'season': {row}")
                continue

            if season not in data_by_season:
                data_by_season[season] = []

            data_by_season[season].append(row)

        # Write separate CSV files for each season
        for season, rows in data_by_season.items():
            output_file = os.path.join(output_dir, f'{season}.csv')
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"Written {len(rows)} rows to {output_file}")



if __name__ == "__main__":
    # Input and output paths
    input_csv = os.path.join("..", "data", "raw", "PGA_Raw_Data_2015_2022.csv")
    output_directory = os.path.join("..", "data", "seasons")
    split_csv_by_season(input_csv, output_directory)
