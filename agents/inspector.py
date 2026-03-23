from openai import OpenAI




class Modelinspector:
    def __init__(self, client: OpenAI):
        self.client = client

    def _inspector(self, code, error_message=None, 
        model="gpt-5.1"):

        if error_message:

            prompt = (
                f"You are a Senior Python Developer. Please evaluate the following Python code and the error message from the last execution attempt. "
                f"This was the original code: {code}"
                f"The previous code failed with the following error:\n"
                f"--- ERROR ---\n{error_message}\n--------------\n"
                f"Please analyze the error and the code, and provide a corrected version. "
                "Only output the corrected code, no explanations, no markdown fences."
                "Make minimal adjustments to the original by only changing the parts of the code that are causing the error, and keep the rest of the code intact. "
            )

        else:
            prompt = (
            "Please evaluate if the following Python code is correct and will run without errors. "
            "If it is correct, do nothing. If it is incorrect, please adapt it so that it runs correctly. Only answer with the code.\n\n"
            f"```python\n{code}\n```"
            "Output ONLY the corrected Python code. No explanations, no markdown fences."
            )

        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.1
        )
        return resp.choices[0].message.content