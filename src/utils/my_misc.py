def pprint_string(s: str, title: [str] = None) -> None:
    """
    Prints string in a custom formatted way.

    Args:
        s (string): The string to be printed
        title (str, optional): Title for the print section.
    """
    print("\n============================================================")
    print(f"{'DATAFRAME' if title is None else title.upper()}")
    print("============================================================")
    print(s)
    print("============================================================\n")
