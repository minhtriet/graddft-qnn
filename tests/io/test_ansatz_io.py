from graddft_qnn.io.ansatz_io import AnsatzIO


def test_list_file_handler(tmp_path):
    test_data = [["cat", "dog"], ["red", "blue"], ["one", "two"]]
    handler = AnsatzIO()
    test_file = tmp_path / "test_output.txt"
    handler.write_to_file(test_file, test_data)
    received_data = handler.read_from_file(f"{test_file}.pkl")

    assert received_data == test_data
