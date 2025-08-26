import h5py

import os
import h5py
import statistics
# directory = 'Final planning results/planning results beta=0.0001-all/'
directory = 'planning baseline+lightweight model/'
# List all files in the directory
files = os.listdir(directory)
methods = ['online_stefan', 'once_stefan', 'once' ,'online','actual']
# Create dictionaries to store cumulative costs for each method
method_costs = {}

for file_name in files:
    file_path = os.path.join(directory, file_name)

    # Check if the file is an HDF5 file
    if file_name.endswith('.h5'):
        with h5py.File(file_path, 'r') as file:
            for method in methods:
                group = file[method]
                expected_cost = group.attrs.get('expected_cost', 0.0)
                actual_cost = group.attrs.get('actual_cost', 0.0)

                if method not in method_costs:
                    method_costs[method] = {'expected_cost': [], 'actual_cost': []}

                method_costs[method]['expected_cost'].append(expected_cost)
                method_costs[method]['actual_cost'].append(actual_cost)

                # Print details for this method
                print(f"\nFile: {file_name}")
                print(f"Method: {method}")
                print(f"  Expected Cost: {expected_cost}")
                print(f"  Actual Cost:   {actual_cost}")

# Calculate and print averages for each method
print("\nAverages for each method across all files:")
for method, costs in method_costs.items():
    avg_expected_cost = sum(costs['expected_cost']) / len(costs['expected_cost'])
    avg_actual_cost = sum(costs['actual_cost']) / len(costs['actual_cost'])

    std_expected_cost = statistics.stdev(costs['expected_cost'])
    std_actual_cost = statistics.stdev(costs['actual_cost'])
    print(f"\nMethod: {method}")
    print(f"  Average Expected Cost: {avg_expected_cost:.4f} (± {std_expected_cost:.4f})")
    print(f"  Average Actual Cost:   {avg_actual_cost:.4f} (± {std_actual_cost:.4f})")
    # print(f"\nMethod: {method}")
    # print(f"  Average Expected Cost: {avg_expected_cost}")
    # print(f"  Average Actual Cost:   {avg_actual_cost}")


# List all files in the directory
# files = os.listdir(directory)
#
# for file_name in files:
#     file_path = os.path.join(directory, file_name)
#
#     # Check if the file is an HDF5 file
#     if file_name.endswith('.h5'):
#         with h5py.File(file_path, 'r') as file:
#             print(f"\nFile: {file_name}")
#             for method in file.keys():
#                 group = file[method]
#                 expected_cost = group.attrs.get('expected_cost', 'N/A')
#                 actual_cost = group.attrs.get('actual_cost', 'N/A')
#
#                 print(f"Method: {method}")
#                 print(f"  Expected Cost: {expected_cost}")
#                 print(f"  Actual Cost:   {actual_cost}")
#     else:
#         print(f"Skipping non-HDF5 file: {file_name}")

# file_path = 'Final planning results/planning results beta=0.0001-all/planning_bold-yeti_best_1000_E_A8.h5'
#
# with h5py.File(file_path, 'r') as file:
#     for method in file.keys():
#         group = file[method]
#         expected_cost = group.attrs.get('expected_cost', 'N/A')
#         actual_cost = group.attrs.get('actual_cost', 'N/A')
#         print(f"\nMethod: {method}")
#         print(f"  Expected Cost: {expected_cost}")
#         print(f"  Actual Cost:   {actual_cost}")

    # for group_name in file.keys():
    #     group = file[group_name]
    #     print(f"\nGroup: {group_name}")
    #
    #     for dataset_name in group.keys():
    #         dataset = group[dataset_name]
    #         print(f"  Dataset: {dataset_name}")
    #         print(f"    Shape: {dataset.shape}")
    #         print(f"    Dtype: {dataset.dtype}")
    #         try:
    #             preview = dataset[:5]  # adjust slice as needed
    #             print(f"    Preview: {preview}")
    #         except Exception as e:
    #             print(f"    Could not preview data: {e}")
