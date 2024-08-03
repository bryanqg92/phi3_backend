import gc
import torch
from src.models.phi3_4k import phi3_model
from src.models.llama2_7b import llama2_7b_model
from src.models.orca_mini import orca_mini

class LoadModel:
    _model = None
    _current_model_type = None

    @classmethod
    def _clear_model(cls):
        if cls._model is not None:
            del cls._model
            cls._model = None
            torch.cuda.empty_cache()  # Libera la memoria de CUDA si se está usando GPU
            gc.collect()  # 
        cls._current_model_type = None

    @classmethod
    def initialize_model(cls, model_type='phi3'):
        if cls._current_model_type != model_type:
            cls._clear_model()
            if model_type == 'phi3':
                cls._model = phi3_model()
            elif model_type == 'llama2_7b':
                cls._model = llama2_7b_model()
            elif model_type == 'orca_mini':
                cls._model = orca_mini()
            else:
                raise ValueError("Invalid model type")
            cls._current_model_type = model_type

    @classmethod
    def change_model(cls, model_type):
        cls.initialize_model(model_type)

    @classmethod
    def get_qa_model(cls):
        if cls._model is None:
            cls.initialize_model()  # Inicializa con el modelo por defecto si aún no está inicializado
            
        return cls._model.getPipeline()

    @classmethod
    def release_model(cls):
        cls._clear_model()

    @classmethod
    def get_model_type(cls):
        return cls._current_model_type