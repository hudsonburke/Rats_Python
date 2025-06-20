
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

import numpy as np
def calculate_cop(force: np.ndarray, moment: np.ndarray, z_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the center of pressure (COP) from force and moment data.
    See Winter Biomechanics 4th edition, page 87.
    Args:
        force (numpy.ndarray): The force data.
        moment (numpy.ndarray): The moment data.
        z_offset (float): The offset in the z-direction.
        
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: The calculated COP data and free moment data.
    """
    cop = np.zeros((force.shape[0], 3))
    cop[:, 0] = (z_offset * force[:, 0] - moment[:, 1]) / force[:, 2]
    cop[:, 1] = (z_offset * force[:, 1] + moment[:, 0]) / force[:, 2]
    cop[:, 2] = z_offset
    
    free_moment = np.zeros_like(moment)
    free_moment[:, 2] = moment[:, 2] + force[:, 0] * cop[:, 1] - force[:, 1] * cop[:, 0]
    return (cop, free_moment)

import numpy as np
import polars as pl
def read_mot(file_path: str) -> tuple[pl.DataFrame, dict]:
    """
    Read a .mot file (OpenSim motion file format)
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089415/Motion+.mot+Files
    
    Args:
        file_path (str): The path to the motion file.
        
    Returns:
        tuple[np.ndarray, dict]: A tuple containing the data array and metadata dictionary
    """
 
    metadata = {}
    
    with open(file_path, 'r') as fid:
        # First line is the name
        metadata['name'] = fid.readline().strip()
        
        # Loop through metadata
        while True:
            line = fid.readline()
            if not line:  # EOF
                break
                
            line = line.strip()
            
            # Check for end of header
            if line == 'endheader':
                break
            elif not line:  # Empty line indicates comments section
                metadata['comments'] = []
                # Read comments until next empty line or endheader
                while True:
                    comment_line = fid.readline()
                    if not comment_line:  # EOF
                        break
                    comment_line = comment_line.strip()
                    if not comment_line or comment_line == 'endheader':
                        if comment_line == 'endheader':
                            break
                        break
                    metadata['comments'].append(comment_line)
                
                if comment_line == 'endheader':
                    break
            else:
                # Parse metadata line (key=value format)
                parts = line.split('=', 1)
                if len(parts) < 2:
                    raise ValueError(f"Invalid metadata line: {line}")
                
                key = parts[0].strip()
                value_str = parts[1].strip()
                
                # Try to convert to number, otherwise keep as string
                try:
                    value = float(value_str)
                    # Convert to int if it's a whole number
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    value = value_str
                
                metadata[key] = value
        
        # Column labels are on the next line after 'endheader'
        labels_line = fid.readline().strip()
        metadata['columnLabels'] = labels_line.split('\t')
        
        # Read the rest of the file as data
        data_lines = []
        for line in fid:
            line = line.strip()
            if line:  # Skip empty lines
                row = [float(x) for x in line.split()]
                data_lines.append(row)
        
        # Convert to numpy array
        if data_lines:
            data = np.array(data_lines)
        else:
            data = np.array([]).reshape(0, len(metadata['columnLabels']))
    
    df = pl.DataFrame(data, schema=metadata['columnLabels'])
    
    return df, metadata