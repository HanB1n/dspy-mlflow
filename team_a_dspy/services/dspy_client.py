import asyncio
import dspy

from collections import defaultdict

from sentence_transformers.model_card import IGNORED_FIELDS

from team_a_dspy.services.chroma_client import ChromaClient
from team_a_dspy.services.es_client import ESClient, SAMPLE_SIZE_PER_DAY
from team_a_dspy.services.config import settings

from team_a_dspy.signatures.schema_interpreter import DataAwareSchemaInterpreter

IGNORED_FIELDS = ["@timestamp", "log", "event", "filename", "filename_path"]
NUMBER_OF_DAYS = 7
MAX_SAMPLES_PER_FIELD = min(35, SAMPLE_SIZE_PER_DAY * NUMBER_OF_DAYS)

class DSPYClient:
    def __init__(self, es_client: ESClient | None = None, chroma_client: ChromaClient | None = None):
        # Initialize service clients
        if es_client:
            self.es_client = es_client
        else:
            self.es_client = ESClient()
        if chroma_client:
            self.chroma_client = chroma_client
        else:
            self.chroma_client = ChromaClient()

        self.lm = dspy.LM(
            base_url=settings.llm_base_url,
            model=f"openai/{settings.llm_model_name}",
            api_key=settings.llm_api_key
        )

        # Load DSPY modules
        self.schema_interpreter = DataAwareSchemaInterpreter()

        self.is_compiled = False
    
    def compile(self):
        # Compile DSPY modules with the LLM
        pass
    
    async def fetch_samples(self):
        samples = await self.es_client.get_last_x_days_samples(days=NUMBER_OF_DAYS)
        return samples
    
    @staticmethod
    def flatten_field(doc: dict, field_samples: dict, prefix: str = ""):
        for key, value in doc.items():
            if key in IGNORED_FIELDS:
                continue
            
            current_field = f"{prefix}{key}"
            if isinstance(value, dict):
                DSPYClient.flatten_field(value, field_samples, prefix=f"{current_field}.")
                continue

            if value is None or str(value).strip() == "":
                continue

            if isinstance(value, list):
                if not value:
                    continue
                preview = value[:3]
                val_str = f"[{', '.join(map(str, preview))}...]" if len(value) > 3 else str(value)
            else:
                val_str = str(value)
            
            if len(field_samples[current_field]) < MAX_SAMPLES_PER_FIELD:
                field_samples[current_field].add(val_str)
                    
    async def interpret_field(self):
        samples = await self.fetch_samples()
        field_samples = defaultdict(set)
        for doc in samples:
            self.flatten_field(doc, field_samples)
        field_types = await self.es_client.flatten_es_mapping()
        with dspy.context(lm=self.lm):
            for field_name, sample_values in field_samples.items():
                field_type = field_types.get(field_name, "unknown")
                interpretation = self.schema_interpreter(field_name=field_name, field_type=field_type, sample_values=list(sample_values))
                print(f"Field: {field_name}\nType: {field_type}\nInterpretation: {interpretation}\n{'-'*40}")
                


async def main():
    dspy_client = DSPYClient(es_client=ESClient(), chroma_client=ChromaClient(dev=True))
    dspy_client.compile()
    await dspy_client.interpret_field()

asyncio.run(main())