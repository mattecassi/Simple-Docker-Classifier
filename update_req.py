def update_file(filename, delimiter):
    with open(filename, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()
        for line in lines:
            updated_line = line.split(delimiter)[0] + '\n'
            file.write(updated_line)
    print(f"File '{filename}' has been updated.")

# Example usage: remove everything after the first ',' in each line of 'example.txt'
update_file('requirements.txt', '@')