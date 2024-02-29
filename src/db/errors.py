class DatasetExistsError(Exception):
    """Exception raised when a dataset already exists in the database."""

    def __init__(self, dataset_name):
        self.resource_name = dataset_name
        super().__init__(
            f"The dataset '{dataset_name}' already exists in the database."
        )


class DatasetDoesNotExistError(Exception):
    """Exception raised when a dataset does not exist in the database."""

    def __init__(self, dataset_name):
        self.resource_name = dataset_name
        super().__init__(
            f"The dataset '{dataset_name}' does not exist in the database."
        )
