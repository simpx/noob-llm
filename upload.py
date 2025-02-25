from huggingface_hub import HfApi
from model import Noob
model_path = 'noob_model'
model = Noob.from_pretrained(model_path)
model.save_pretrained(model_path)
api = HfApi()
api.upload_folder(
    folder_path=model_path,
    repo_id="simpx/noob",
    repo_type="model"
)