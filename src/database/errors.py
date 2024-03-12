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


class DocumentDoesNotExistError(Exception):
    """Exception raised when a document does not exist in the database."""

    def __init__(self, dataset, document_id):
        self.resource_name = f"{dataset}/{document_id}"
        super().__init__(
            f"The document '{document_id}' does not exist in the dataset '{dataset}'."
        )


class DocumentExistsError(Exception):
    """Exception raised when a document already exists in the database."""

    def __init__(self, dataset, document_id):
        self.resource_name = f"{dataset}/{document_id}"
        super().__init__(
            f"The document '{document_id}' already exists in the dataset '{dataset}'."
        )


class FileDoesNotExistError(Exception):
    """Exception raised when a file does not exist in the database."""

    def __init__(self, dataset, file_id):
        self.resource_name = f"{dataset}/{file_id}"
        super().__init__(
            f"The file '{file_id}' does not exist in the dataset '{dataset}'."
        )
