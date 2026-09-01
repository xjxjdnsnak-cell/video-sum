"""Shared filename sanitization for export paths."""


def sanitize_filename(title: str) -> str:
    """Sanitize a video title for safe use as a file name.

    Keeps alphanumeric characters and " -_", replaces everything else
    (including path separators, ".." and ":") with "_", and truncates
    the result to 50 characters.
    """
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:50]
