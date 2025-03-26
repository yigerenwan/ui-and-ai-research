import os
import openai
from datetime import datetime
from typing import Optional
import json

class CodeGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CodeGenerator with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Ensure the output directory exists
        self.output_dir = "/tmp/python-use-agent"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_code(self, prompt: str) -> dict:
        """
        Generate Python code based on the prompt using GPT-4.
        
        Args:
            prompt: Description of the code to generate
            
        Returns:
            dict containing:
                'success': bool indicating if generation was successful
                'file_path': path to the saved Python file
                'code': generated code
                'error': error message if generation failed
        """
        try:
            # Construct the system message to ensure Python code output
            messages = [
                {"role": "system", "content": "You are a Python programming expert. Respond only with Python code, no explanations. The code should be complete and runnable."},
                {"role": "user", "content": prompt}
            ]

            # Generate code using GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4",  # or use specific model version as needed
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            # Extract the generated code
            generated_code = response.choices[0].message.content.strip()

            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.py"
            file_path = os.path.join(self.output_dir, filename)

            # Save the code to file
            with open(file_path, 'w') as f:
                f.write(generated_code)

            return {
                'success': True,
                'file_path': file_path,
                'code': generated_code,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'file_path': None,
                'code': None,
                'error': str(e)
            }

    def generate_and_execute(self, prompt: str) -> dict:
        """
        Generate and execute Python code based on the prompt.
        
        Args:
            prompt: Description of the code to generate
            
        Returns:
            dict containing generation and execution results
        """
        # First generate the code
        generation_result = self.generate_code(prompt)
        
        if not generation_result['success']:
            return generation_result

        # Import the execution function
        from exe import execute_python_file
        
        # Execute the generated code
        execution_result = execute_python_file(generation_result['file_path'])
        
        # Combine the results
        return {
            'generation': generation_result,
            'execution': execution_result
        }

# Example usage
if __name__ == "__main__":
    generator = CodeGenerator()
    
    # Example prompt
    prompt = "Write a Python script that calculates the fibonacci sequence up to the 10th number"
    
    # Generate and execute the code
    result = generator.generate_and_execute(prompt)
    print(json.dumps(result, indent=2))
