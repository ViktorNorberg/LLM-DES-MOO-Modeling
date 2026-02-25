import subprocess
import os
import shutil

def mermaid(mmd_path, output_path):
    if not os.path.exists(mmd_path):
        raise FileNotFoundError(f".mmd file not found: {mmd_path}")

    if shutil.which("npx") is None:
        raise RuntimeError("npx not found. Install Node.js to use this renderer.")

    # The --yes flag ensures npx doesn't stop to ask permission to install the package
    cmd = [
        "npx", "--yes",
        "-p", "@mermaid-js/mermaid-cli", "mmdc",
        "-i", mmd_path,
        "-o", output_path
    ]

    print(f"Rendering Mermaid diagram: {mmd_path} → {output_path}")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print("Rendering complete. Check your folder for the PNG!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during rendering: {e}")

# --- COMPLETION CODE ---

# 1. Define paths
mmd_test_file = "test_diagram.mmd"
png_output_file = "test_diagram.png"

# 2. Create a simple Mermaid diagram content
test_content = """
graph TD
    A[Install Node.js] --> B{Is npx working?}
    B -- Yes --> C[Success!]
    B -- No --> D[Check PATH]
"""

# 3. Write the .mmd file to disk
with open(mmd_test_file, "w") as f:
    f.write(test_content)

# 4. Run the renderer

mermaid(mmd_test_file, png_output_file)