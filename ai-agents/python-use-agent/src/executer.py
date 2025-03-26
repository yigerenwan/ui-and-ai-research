import sys
import subprocess
from typing import Dict, Any

def execute_python_file(file_path: str) -> Dict[str, Any]:
    """
    Execute a Python file and return the results.
    
    Args:
        file_path (str): Path to the Python file to execute
        
    Returns:
        Dict containing:
            'success': bool indicating if execution was successful
            'output': stdout/stderr output from the execution
            'error': error message if execution failed
    """
    try:
        # Run the Python file in a subprocess
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for safety
        )
        
        if result.returncode == 0:
            return {
                'success': True,
                'output': result.stdout,
                'error': None
            }
        else:
            return {
                'success': False,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': None,
            'error': 'Execution timed out after 30 seconds'
        }
    except Exception as e:
        return {
            'success': False,
            'output': None,
            'error': str(e)
        }

# Example usage
if __name__ == "__main__":
    # Example: execute a file in /tmp
    test_file = "/tmp/test.py"
    result = execute_python_file(test_file)
    print(f"Execution result: {result}")
