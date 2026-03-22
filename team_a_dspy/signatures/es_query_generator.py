import dspy

from services.chroma_client import ChromaClient
from signatures.schema_interpreter import SchemaRetriever

class NLToQueryDSL(dspy.Module):
    def __init__(self, chroma_client: ChromaClient):
        super().__init__()
        self.chroma_client = chroma_client
        self.generate_query = dspy.ChainOfThought(NLToQuerySignature)
        self.schema_retriever = SchemaRetriever(chroma_client=chroma_client)

    def forward(self, nl_query: str) -> dict:
        es_schema = self.schema_retriever(nl_query=nl_query)
        generated_query = self.generate_query(nl_query=nl_query, es_schema=es_schema)
        print(f"Reasoning trace for query generation:{generated_query.reasoning}")
        return generated_query.query_dsl
    
class NLToQuerySignature(dspy.Signature):
    """
    Convert a natural language query into an Elasticsearch Query DSL format.
    This module takes a natural language query as input and generates a corresponding Elasticsearch Query DSL JSON object.
    Use the ES schema of the GDELT index to inform the generation of the Query DSL, ensuring that field names and types are correctly referenced in the output.
    If the natural language query references requirements that cannot be fulfilled based on the ES schema, rely on the existing fields in the schema to generate the most relevant Query DSL possible, and do not include any fields that are not present in the ES schema.
    Do not include any fields in the output that are not present in the ES schema. 
    Do not use any scripts.
    The output should be a valid Elasticsearch Query DSL that can be directly used to query the GDELT index.
    """
    nl_query: str = dspy.InputField(desc="A natural language query describing the search criteria.")
    es_schema: dict = dspy.InputField(desc="The Elasticsearch schema of the GDELT index, including field names, types and descriptions.")
    
    query_dsl: dict = dspy.OutputField(
        desc="A JSON object representing the equivalent Elasticsearch Query DSL for the given natural language query."
    )