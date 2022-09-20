import datetime
import textwrap
import warnings


def formatwarning(message, category, filename, lineno, _=None):

    warning = ""
    formatted_lines = textwrap.wrap(f"{category.__name__}: {message}", 118)
    warning += f"[{datetime.datetime.now()}] {filename}:{lineno}\n"
    for formatted_line in formatted_lines:
        warning += f"  {formatted_line}\n"

    return warning


def showwarning(message, category, filename, lineno, file=None, _=None):

    if file is None:
        file = open("warnings.log", "w")

    warning = formatwarning(message, category, filename, lineno)
    file.write(warning)
    file.close()


if __name__ == "scripts":

    warnings.formatwarning = formatwarning
    warnings.showwarning = showwarning
