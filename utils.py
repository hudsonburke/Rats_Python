
def snake_to_pascal(snake_str : str) -> str:
    return ''.join(word.capitalize() for word in snake_str.split('_'))

def pascal_to_snake(pascal_str : str) -> str:
    return ''.join(['_' + c.lower() if c.isupper() else c for c in pascal_str]).lstrip('_')

def c3d_scan(root_directory: str) -> list[str]:
    """
    Scan the given root directory for C3D files and return a list of their paths.
    
    Args:
        root_directory (str): The root directory to scan for C3D files.
        
    Returns:
        list[str]: A list of paths to C3D files found in the directory.
    """
    import os
    c3d_files = []
    with os.scandir(root_directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.c3d'):
                c3d_files.append(entry.path)
            elif entry.is_dir():
                c3d_files.extend(c3d_scan(entry.path))
    return c3d_files

from typing import Generator
def c3d_scan_gen(root_directory: str) -> Generator[str, None, None]:
    """
    Generator to scan the given root directory for C3D files.
    
    Args:
        root_directory (str): The root directory to scan for C3D files.
        
    Yields:
        str: Paths to C3D files found in the directory.
    """
    import os
    with os.scandir(root_directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.c3d'):
                yield entry.path
            elif entry.is_dir():
                yield from c3d_scan_gen(entry.path)
                
import os
import re 
def scandir_regex(directory, pattern):
    """
    Iterates through directory entries and yields names that match the pattern.
    """
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and re.match(pattern, os.path.normpath(entry.path)):
                yield entry.path
            elif entry.is_dir():
                yield from scandir_regex(entry.path, pattern)


                
