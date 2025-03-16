class AnsatzIO:
    def __init__(self):
        pass

    @staticmethod
    def write_to_file(file_path, data):
        with open(file_path, "w") as file:
            for sublist in data:
                file.write(",".join(sublist) + "\n")

    @staticmethod
    def read_from_file(file_path):
        data = []
        with open(file_path) as file:
            for line in file:
                data.append(line.strip().split(","))
        return data
