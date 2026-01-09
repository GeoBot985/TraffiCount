# collector.py
# Run this from the project root: python -m Autonomous_Reasoning_System.collector

import os
import io
import sys

def consolidate_python_files(output_filename='consolidated_app.py', start_path='.'):
    script_name = os.path.basename(__file__)
    abs_start = os.path.abspath(start_path)
    print(f"Consolidating from: {abs_start}")
    print(f"Output file: {output_filename}")
    print()

    files_collected = 0

    with io.open(output_filename, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(start_path):
            # Skip junk dirs
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            if 'venv' in dirs:
                dirs.remove('venv')
            if '.git' in dirs:
                dirs.remove('.git')

            py_files = [f for f in files if f.endswith('.py')]

            for filename in py_files:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, start_path)

                # Skip self and output file
                if filename == script_name or filename == output_filename:
                    print(f"Skipping: {rel_path}")
                    continue

                try:
                    with io.open(full_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()

                    header = f"\n# {'='*75}\n"
                    header += f"# FILE START: {rel_path}\n"
                    header += f"# {'='*75}\n\n"

                    outfile.write(header)
                    outfile.write(content)
                    outfile.write('\n\n')

                    files_collected += 1
                    print(f"Added: {rel_path}")

                except Exception as e:
                    print(f"ERROR reading {rel_path}: {e}")

    print(f"\nDone! {files_collected} files → {output_filename}")

if __name__ == '__main__':
    # When run as module (python -m Autonomous_Reasoning_System.collector)
    # __file__ is not defined the same way → fallback to current dir
    consolidate_python_files()