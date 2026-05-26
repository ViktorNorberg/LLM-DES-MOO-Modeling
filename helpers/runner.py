import subprocess, sys, tempfile, textwrap
from pathlib import Path

def run_python_code(code_str: str, timeout=40000):

    # Create temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", 
                                         delete=False, 
                                         suffix=".py", 
                                         encoding="utf-8") as tmp:
            tmp.write(textwrap.dedent(code_str))
            tmp_path = Path(tmp.name)

        # Run the process
        # Use sys.executable for same virtual environment

        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check for execution errors
        if result.returncode != 0:
            raise RuntimeError(f"Execution Error (Code {result.returncode}):\n{result.stderr}")

        # Return the the KPIs
        return result.stdout

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Optimization timed out after {timeout} seconds.")
    
    finally:
        # Cleanup: Delete the file regardless of success/failure
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass # Prevent cleanup errors from crashing the main script

