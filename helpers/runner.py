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

def run_python_code(code_str: str, timeout=40000):
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


"""
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

def run_MOO_code(code_str: str, timeout=15000):
    tmp_path = None
    captured_output = [] # We will store lines here to return them at the end
    
    try:
        # 1. Create the temp file (same as before)
        with tempfile.NamedTemporaryFile(mode="w", 
                                         delete=False, 
                                         suffix=".py", 
                                         encoding="utf-8") as tmp:
            tmp.write(textwrap.dedent(code_str))
            tmp_path = Path(tmp.name)

        # 2. Start the process using Popen instead of run
        # stdout=PIPE allows us to "tap into" the output stream
        process = subprocess.Popen(
            [sys.executable, "-u", str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Line buffered
        )

        # 3. Read the output in real-time
        # This loop runs while the simulation is running
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end="", flush=True) # Send to your terminal
                captured_output.append(line)    # Save for the return value

        # Wait for the process to finish and get stderr
        stdout_rem, stderr = process.communicate(timeout=timeout)
        
        # Catch any remaining stdout after the loop
        if stdout_rem:
            print(stdout_rem, end="", flush=True)
            captured_output.append(stdout_rem)

        # 4. Check for execution errors
        if process.returncode != 0:
            raise RuntimeError(f"Execution Error (Code {process.returncode}):\n{stderr}")

        # 5. Return the full string (concatenating the list of lines)
        return "".join(captured_output)

    except subprocess.TimeoutExpired:
        # If it times out, we must kill the process manually
        process.kill()
        raise RuntimeError(f"Optimization timed out after {timeout} seconds.")
    
    finally:
        # 6. Cleanup (same as before)
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

"""