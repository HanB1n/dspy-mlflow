import dspy

class SchemaInterpreter(dspy.Signature):
    """
    Analyze a GDELT database field based on its name, type, and real sample values from the last 7 days.
    Provide a clear, concise natural language interpretation of what this field represents 
    and how it should be used in an Elasticsearch query.
    """
    field_name: str = dspy.InputField(desc="Name of the field in the GDELT schema.")
    field_type: str = dspy.InputField(desc="Elasticsearch data type of the field.")
    sample_values: str = dspy.InputField(desc="A list of real values found in this field over the last 7 days.")
    
    interpretation: str = dspy.OutputField(
        desc="A 1-2 sentence explanation of the field and its semantic meaning."
    )

class DataAwareSchemaInterpreter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.interpret = dspy.Predict(SchemaInterpreter)
        
    def forward(self, field_name, field_type, sample_values):
        return self.interpret(field_name=field_name, field_type=field_type, sample_values=sample_values)