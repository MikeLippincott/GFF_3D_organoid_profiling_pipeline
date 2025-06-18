class ProcessorTypeError(Exception):
    print("""
    Processor type not recognized. Use 'CPU' or 'GPU'.
    Though the dev on this project really should not allow for the code to reach this point
    without the validation of a valid processor type.
    My apologies for the oversight.
    """)
