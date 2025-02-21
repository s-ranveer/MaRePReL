import os
import csv


def reverse_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

def rename_columns_in_csv(directory, mapping):
    # Recursively search for CSV files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                print("Processing:", csv_file)
                rev_mapping = reverse_dict(mapping)
                # Read the CSV file and rename columns based on the mapping
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames
                    renamed_columns = [mapping.get(column, column) for column in columns]
                    
    
                    # Write to a new CSV file with renamed columns
                    temp_file = csv_file + '.tmp'
                    with open(temp_file, 'w', newline='') as new_f:
                        writer = csv.DictWriter(new_f, fieldnames=renamed_columns)
                        writer.writeheader()
                        filtered_row = {}
                        # print(mapping)
                        for row in reader:
                            for key in row:
                           

                                    if key in renamed_columns:    
                                        filtered_row[key] = row[key] 
                                    else:
                                        filtered_row[mapping[key]] = row[key]
                               
                        
                            writer.writerow(filtered_row)
                    
                    # Rename the temporary file to the original CSV file name
                    os.replace(temp_file, csv_file)
                    print("Done")

# Example usage:
directory = "logs/MultiAgentTaxi/PS_DQN/DQN_rainbow_config_v2/2T2P"
mapping = {"custom_metrics/Success_mean": "custom_metrics/success_mean", "custom_metrics/Episode Return_mean": "custom_metrics/episode_reward_mean"}


rename_columns_in_csv(directory, mapping)
