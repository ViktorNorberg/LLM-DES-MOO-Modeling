from openai import OpenAI

class Modelvisualizer:
    def __init__(self, client: OpenAI):
        self.client = client

    def visualize_agent(self, model_source: str) -> str:
        print("\nVisualizer activated:")
        return self._visualize_agent(model_source)

    def _visualize_agent(
        self,
        model_source: str,
        model: str = "gpt-5-mini",
    ) -> str:
        prompt = (
            "You get Python code for a discrete-event simulation model.\n"
            "Create a Mermaid flowchart that shows the main material/part flow.\n\n"
            "HARD REQUIREMENTS:\n"
            "1) Output ONLY Mermaid code: no markdown fences, no ```python blocks, no explanations.\n"
            "2) The first non-empty line MUST be exactly: flowchart TD\n"
            "3) Immediately after that, define these classes and use them consistently:\n"
            "   classDef buffer fill:#ffffff,stroke:#333333,stroke-width:1px,stroke-dasharray:3 3,color:#000;\n"
            "   classDef machine fill:#d2e7ff,stroke:#004a99,stroke-width:1px,color:#000;\n"
            "   classDef store fill:#ffe08a,stroke:#b87a00,stroke-width:1px,color:#000;\n"
            "   classDef sink fill:#ffb3b3,stroke:#990000,stroke-width:1px,color:#000;\n"
            "   classDef defect fill:#ff9999,stroke:#660000,stroke-width:1px,color:#000;\n"
            "   classDef helper fill:#e0e0e0,stroke:#666666,stroke-width:1px,color:#000;\n\n"
            "Mapping:\n"
            "- Raw material inputs / Stores / warehouses => class 'store'.\n"
            "- Finished-good sinks => class 'sink'.\n"
            "- Defect / scrap sinks => class 'defect'.\n"
            "- Machines, robots, presses, cells => class 'machine'.\n"
            "- Buffers, queues, delay buffers => class 'buffer'.\n"
            "- Splitters, mergers, routers, helper logic => class 'helper'.\n\n"
            "Label format:\n"
            "- Use valid identifiers (letters, digits, underscore) for node IDs.\n"
            "- ALL node labels must use real line breaks inside the brackets.\n"
            "- Never output '\\n' or '\\\\n' anywhere in any label (including stores and sinks).\n"
            "- The first line of every label must be the node name written in **bold**, using Markdown syntax.\n"
            "  Example:\n"
            "      node_id[\"**Label**\n"
            "      CT=12s\n"
            "      AVB=90%\n"
            "      MTTR=68s\"]:::machine\n"
            "Edges:\n"
            "- Use '-->' to show material flow, top to bottom.\n"
            "- Add short edge labels where routing is important (optional).\n\n"
            "Here is the Python model:\n\n"
            f"{model_source}\n\n"
            "Now output only the Mermaid flowchart that follows these rules."
        )
        resp = self.client.chat.completions.create(
            model=model,messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def _generate_MOO_UML(self, model_code: str, objectives, input_variables,
        model: str ="gpt-5-mini") -> str:
        prompt = (
                "You are an AI assistant that generates UML activity diagrams from natural language."
                "The UML should include the classes, their attributes, and methods. "
                "Focus on the parts of the code that are relevant to the following objectives, input variables, and output variables.\n\n"
                "The provided python code is of a simulation model of a production system"
                "Please generate a UML digram of an MOO algorithm that will optimize the following simulation"
                "Your UML diagram should be focused on how the MOO algorithm will interact with the existing code, and how it will optimize it. "
                f"Here is my Python code:\n\n```python\n {model_code}\n```\n\n"
                f"MOO objectives: {objectives}\n"
                f"Input Variables: {input_variables}\n"
                "Keep the UML diagram simple and understandable for a human production manager"

                 "HARD REQUIREMENTS:\n"
                "1) Output ONLY Mermaid code: no markdown fences, no ```python blocks, no explanations.\n"
                "2) The first non-empty line MUST be exactly: flowchart TD\n"
                "Only answer with the UML diagram in a mermaid code format."
                "3) IMPORTANT: Never use square brackets [ ] or parentheses ( ) inside a node label. "
                "Use curly braces { } or just plain text for ranges (e.g., {1-10} instead of [1,10]).\n"
                "4) Every node must be formatted as: NodeID[\"**Node Name**<br/>Description\"]"

                "Label format:\n"
                "- Use valid identifiers (letters, digits, underscore) for node IDs.\n"
                "- ALL node labels must use real line breaks inside the brackets.\n"
                "- Never output '\\n' or '\\\\n' anywhere in any label\n"
                "- The first line of every label must be the node name written in **bold**, using Markdown syntax.\n"
                )

        resp = self.client.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}]) 
            #temperature=0.1)
        try:
            return resp.choices[0].message.content
        except Exception as e:
            raise