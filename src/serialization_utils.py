import pickle


def serialize_to_file(data, filename):
    """
    Serialize the given data and save it to a file using pickle.

    Args:
        data: The data to be serialized.
        filename (str): The name of the file to save the serialized data.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def deserialize_from_file(filename):
    """
    Load and deserialize data from a file using pickle.

    Args:
        filename (str): The name of the file containing the serialized data.

    Returns:
        The deserialized data.
    """
    with open(filename, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
