
#!/usr/bin/env python3
"""
Script to collect all .py and .ipynb files from a directory tree
and output them to a single text file with directory annotations.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def read_ipynb_content(filepath):
    """Read and extract code from Jupyter notebook files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        code_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code_cells.append(''.join(source))
                else:
                    code_cells.append(source)
            elif cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                if isinstance(source, list):
                    markdown_text = ''.join(source)
                else:
                    markdown_text = source
                # Add markdown cells as comments
                commented_markdown = '\n'.join(f'# {line}' for line in markdown_text.split('\n'))
                code_cells.append(f'\n# [Markdown Cell]\n{commented_markdown}\n')
        
        return '\n\n'.join(code_cells)
    except Exception as e:
        return f"# Error reading notebook: {str(e)}"


def read_py_content(filepath):
    """Read content from Python files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Error reading file: {str(e)}"


def collect_python_files(root_dir, output_file, include_hidden=False):
    """
    Collect all .py and .ipynb files and write them to output file.
    
    Args:
        root_dir: Root directory to search for files
        output_file: Path to output text file
        include_hidden: Whether to include files in hidden directories
    """
    root_path = Path(root_dir).resolve()
    output_path = Path(output_file).resolve()
    
    # Collect all relevant files
    py_files = []
    ipynb_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip hidden directories if requested
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        dir_path = Path(dirpath)
        
        # Skip if we're in a hidden directory
        if not include_hidden and any(part.startswith('.') for part in dir_path.parts):
            continue
        
        for filename in filenames:
            if filename.endswith('.py'):
                py_files.append(dir_path / filename)
            elif filename.endswith('.ipynb') and not filename.startswith('.'):
                ipynb_files.append(dir_path / filename)
    
    # Sort files for consistent output
    all_files = sorted(py_files + ipynb_files)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as out:
        # Write header
        out.write(f"# Python and Jupyter Notebook Files Export\n")
        out.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"# Root directory: {root_path}\n")
        out.write(f"# Total files: {len(all_files)}\n")
        out.write("=" * 80 + "\n\n")
        
        # Write file contents
        for i, filepath in enumerate(all_files, 1):
            # Get relative path from root directory
            rel_path = filepath.relative_to(root_path)
            
            # Write file header
            out.write(f"\n{'=' * 80}\n")
            out.write(f"# File {i}/{len(all_files)}: {rel_path}\n")
            out.write(f"# Full path: {filepath}\n")
            out.write(f"# File type: {'Python' if filepath.suffix == '.py' else 'Jupyter Notebook'}\n")
            out.write(f"{'=' * 80}\n\n")
            
            # Write file content
            if filepath.suffix == '.py':
                content = read_py_content(filepath)
            else:  # .ipynb
                content = read_ipynb_content(filepath)
            
            out.write(content)
            out.write(f"\n\n{'=' * 80}\n")
            out.write(f"# End of file: {rel_path}\n")
            out.write(f"{'=' * 80}\n\n")
    
    print(f"Successfully exported {len(all_files)} files to {output_path}")
    print(f"  - Python files: {len(py_files)}")
    print(f"  - Jupyter notebooks: {len(ipynb_files)}")


def main():
    parser = argparse.ArgumentParser(
        description='Export all .py and .ipynb files from a directory to a single text file'
    )
    parser.add_argument(
        'directory',
        help='Directory to search for Python and Jupyter notebook files'
    )
    parser.add_argument(
        '-o', '--output',
        default='python_files_export.txt',
        help='Output text file name (default: python_files_export.txt)'
    )
    parser.add_argument(
        '--include-hidden',
        action='store_true',
        help='Include files in hidden directories (starting with .)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        return 1
    
    # Run the collection
    try:
        collect_python_files(args.directory, args.output, args.include_hidden)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())