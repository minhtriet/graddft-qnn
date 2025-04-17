import pickle


class AnsatzIO:
    def __init__(self):
        pass

    @staticmethod
    def write_to_file(file_path, data):
        with open(f"{file_path}.pkl", "wb") as f:
            pickle.dump(data, f)
        with open(file_path, "w") as file:
            for sublist in data:
                file.write(str(sublist) + "\n")

    @staticmethod
    def read_from_file(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
