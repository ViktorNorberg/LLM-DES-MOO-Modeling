from openai import OpenAI




class Modelinspector:
    def __init__(self, client: OpenAI):
        self.client = client

    def _inspect(self, code, error_message,
        model="gpt-5.1"):

        prompt = (
            f"You are a Software Developer. Please evaluate the following code and the error message from the last execution attempt. "
            f"This was the original code: {code}"
            f"The previous code failed with the following error:\n"
            f"--- ERROR ---\n{error_message}\n--------------\n"
            f"Please analyze the error and the code, and provide a corrected version. "
            "Only output the corrected code, no explanations, no markdown fences."
            "Make minimal adjustments to the original by only changing the parts of the code that are causing the error, and keep the rest of the code intact. "
        )

        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.1
        )
        return resp.choices[0].message.content