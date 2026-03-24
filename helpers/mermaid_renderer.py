import subprocess
import os
import shutil
from sys import path
from agents.inspector import Modelinspector
from helpers.other_helpers import remove_code_wrappers, save_model, run_python_code
from pathlib import Path



def render_mermaid_to_png(mmd_path, output_path, client):
    if not os.path.exists(mmd_path):
        raise FileNotFoundError(f".mmd file not found: {mmd_path}")

    if shutil.which("npx") is None:
        raise RuntimeError("npx not found. Install Node.js to use this renderer.")
    
    max_attempts = 3
    attempt = 0
    inspector = Modelinspector(client)

    with open(mmd_path, "r", encoding='utf-8') as f:
        code = f.read()

    while attempt < max_attempts:
        print(f"Inspecting mermaid code (Attempt {attempt + 1})...")

        if attempt > 0:
            # pass the error_message if it exists
            code = inspector._inspect(code, error_message)
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(code)
            
        try:
    
            cmd = [
                "npx", "--yes",
                "@mermaid-js/mermaid-cli",
                "-i", mmd_path,
                "-o", output_path
            ]

            print(f"Rendering Mermaid diagram: {mmd_path} → {output_path}")
            subprocess.run(cmd, check=True, shell=True)
            print("Rendering complete.")
            break

        except Exception as e:
            error_message = str(e)
            print(f"Attempt {attempt + 1} failed, this is the error message:\n\n {error_message}\n\n, repairing code...")
            attempt += 1
            if attempt == max_attempts:
                print("Maximum fix attempts reached. Please fix the code manually.")
                return None