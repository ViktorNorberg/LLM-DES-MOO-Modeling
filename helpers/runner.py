import subprocess, sys, tempfile, textwrap
from pathlib import Path

"""
def run_python_code(code_str: str, timeout = 600):
    # Write the generated code to a scratch file
    with tempfile.NamedTemporaryFile(mode="w",
                                     delete=False,
                                     suffix=".py",
                                     encoding="utf-8") as tmp:
        tmp.write(textwrap.dedent(code_str))
        tmp_path: Path = Path(tmp.name)

    # Launch a new interpreter so the code runs in isolation
    result = subprocess.run(
        [sys.executable, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=timeout
    )
    # check if things go well and delete the file if it does.
    if result.returncode == 0:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    if result.returncode != 0:
        raise RuntimeError(
            f"{tmp_path} exited {result.returncode}:\n{result.stderr}")
    return result.stdout

"""

def run_python_code(code_str: str, timeout=3600):
    # 1. Create the temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", 
                                         delete=False, 
                                         suffix=".py", 
                                         encoding="utf-8") as tmp:
            tmp.write(textwrap.dedent(code_str))
            tmp_path = Path(tmp.name)

        # 2. Run the process
        # We use sys.executable to ensure we use the SAME virtual environment
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # 3. Check for execution errors
        if result.returncode != 0:
            # We raise an error that contains the full traceback from stderr
            raise RuntimeError(f"Execution Error (Code {result.returncode}):\n{result.stderr}")

        # 4. Return the standard output (your KPIs and tables)
        return result.stdout

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Optimization timed out after {timeout} seconds.")
    
    finally:
        # 5. Cleanup: Always delete the file if it exists, regardless of success/failure
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass # Prevent cleanup errors from crashing the main script