import torch
import transformers
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

class llama2_7b_model():
    
    def __init__(self) -> None:
        
        self.model_hf = "meta-llama/Llama-2-7b-chat-hf"
        self.token = "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"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf, padding_side='left', token = self.token)
        self.pipeline = pipeline(
            "text-generation",            # Tipo de tarea: generación de texto
            model=self.model_hf,             # Nombre del modelo preentrenado
            token=self.token,
            tokenizer=self.tokenizer,          # Tokenizador a utilizar
            torch_dtype=torch.bfloat16,   # Tipo de datos a utilizar en PyTorch (bfloat16 para mejor eficiencia)
            trust_remote_code=True,       # Permite el uso de código remoto confiable
            device_map="auto",            # Asigna automáticamente los dispositivos para la ejecución (CPU/GPU)

            model_kwargs = {
                'temperature': 0.3,       # Controla la aleatoriedad de las predicciones (0.5 para un equilibrio entre coherencia y creatividad)
                'max_length': 1500,       # Longitud máxima del texto generado
                'do_sample': True,        # Activa la muestreo aleatorio durante la generación de texto
                'top_k': 5,               # Limita la consideración de las 10 mejores opciones para cada token (mejora la calidad)
                'eos_token_id': self.tokenizer.eos_token_id,  # Token de fin de secuencia para terminar la generación
                'attn_implementation': 'eager',
                'cache_dir':"offload_models/phi3"
            }
        )
    
    def getPipeline(self) -> HuggingFacePipeline:

        llm = HuggingFacePipeline(
            pipeline = self.pipeline,
            )
        return llm
