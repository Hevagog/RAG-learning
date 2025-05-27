dataset = []

def load_dataset(path: str) -> list:
    """
    Load the dataset from a file.
    """
    with open(path, 'r') as file:
        data = file.readlines()
    return data