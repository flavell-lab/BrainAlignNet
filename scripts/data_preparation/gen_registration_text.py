import os
import itertools

filepath = '/scratch/nar8991/computer_vision/data/raw_data/2022-01-06-01-SWF360_NRRD_cropped'
registration_file = 'registration_problems.txt'
save_file = os.path.join(filepath, registration_file)

registration_problems = set()

for root, dirs, files in os.walk(filepath):
    for file in files:
        # if file.endswith('.nrrd'):
        if file.endswith('_ch1.nrrd'):
            t_stamp = file.split('_')[1]
            # Cut off the leftmost character
            s_modified = t_stamp[1:]

            # Convert the remaining string to integer
            result = int(s_modified)

            registration_problems.add(result)

print(len(registration_problems))

# Convert registration problems to an ordered list
registration_problems = sorted(registration_problems)

# Create combinations of all registration problem values
all_combinations = []
val = itertools.combinations(registration_problems, 2)
all_combinations.extend(val)
print(len(all_combinations))


with open(save_file, 'w') as file:
    for combo in all_combinations:
        file.write(' '.join(map(str, combo)) + '\n')

# Check out what ch1 and ch2 are eventually