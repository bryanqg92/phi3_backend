import torch
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

class orca_mini():

    def __init__(self) -> None:
        
        self.model_hf = "pankajmathur/orca_mini_3b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf, padding_side='left')
        self.pipeline = pipeline(
            "text-generation",            # Tipo de tarea: generación de texto
            model=self.model_hf,             # Nombre del modelo preentrenado
            tokenizer=self.tokenizer,          # Tokenizador a utilizar
            torch_dtype=torch.bfloat16,   # Tipo de datos a utilizar en PyTorch (bfloat16 para mejor eficiencia)
            trust_remote_code=True,       # Permite el uso de código remoto confiable
            device_map="auto",            # Asigna automáticamente los dispositivos para la ejecución (CPU/GPU)


            # Parametros para el modelo
            model_kwargs = {
                'temperature': 0.3,       # Controla la aleatoriedad de las predicciones (0.5 para un equilibrio entre coherencia y creatividad)
                'max_length': 1500,       # Longitud máxima del texto generado
                'do_sample': True,        # Activa la muestreo aleatorio durante la generación de texto
                'top_k': 5,              # Limita la consideración de las 10 mejores opciones para cada token (mejora la calidad)
                'eos_token_id': self.tokenizer.eos_token_id,  # Token de fin de secuencia para terminar la generación
                'attn_implementation': 'eager',
                'cache_dir':"offload_models/orca_mini"
            }
        )
    
    def getPipeline(self) -> HuggingFacePipeline:

        llm_phi3 = HuggingFacePipeline(
            pipeline = self.pipeline,
            )
        return llm_phi3

        