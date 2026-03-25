from services.es_client import ESClient
from services.config import settings
from elasticsearch import BadRequestError

class SandboxESClient(ESClient):
    """
    Client for interacting with the sandbox Elasticsearch instance.
    This is a subclass of ESClient configured to connect to the sandbox ES instance using settings from the config.
    """
    def __init__(self):
        super().__init__(settings.sandbox_es_host, settings.sandbox_es_username, settings.sandbox_es_password, settings.sandbox_es_index, settings.sandbox_es_verify_ssl)

    def validate_query_dsl(self, query_dsl: dict):
        """
        Validates the generated Query DSL by executing it against the sandbox ES instance and checking for errors.
        
        Args:
            query_dsl (dict): The generated Elasticsearch Query DSL to be validated.
                Can be either wrapped as {"query_dsl": {...}} or raw DSL directly.
        Returns:
            dict: A dictionary containing the validation results, including whether the query is valid and any feedback or error messages from Elasticsearch.
        """
        # Handle both wrapped ("query_dsl" key) and unwrapped (raw DSL) formats
        if "query_dsl" in query_dsl:
            query = query_dsl["query_dsl"]
        else:
            query = query_dsl
        
        try:
            response = self.es.search(index=self.index, body=query)
            failed_shards = response.body.get("_shards", {}).get("failed", 0)
            if failed_shards:
                failures = response.body.get("_shards", {}).get("failures", [])
                return {
                    "is_valid": False,
                    "feedback": f"Shard failures detected: {failures}",
                }
            return {
                "is_valid": True,
                "feedback": "Query executed successfully in sandbox Elasticsearch.",
            }
        except BadRequestError as exc:
            error_text = str(exc)
            return {
                "is_valid": False,
                "feedback": error_text,
            }
        except Exception as exc:
            return {
                "is_valid": False,
                "feedback": f"Unexpected validation error: {exc}",
            }
