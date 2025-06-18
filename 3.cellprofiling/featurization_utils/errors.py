class ProcessorTypeError(Exception):
    """
    Exception raised when an unrecognized processor type is encountered.
    Use 'CPU' or 'GPU' as valid processor types.
    """

    def __str__(self):
        return (
            "Processor type not recognized. Use 'CPU' or 'GPU'. "
            "Though the dev on this project really should not allow for the code to reach this point "
            "without the validation of a valid processor type. My apologies for the oversight."
        )
