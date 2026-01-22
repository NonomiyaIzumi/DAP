import argparse
import yaml
import torch
from types import SimpleNamespace
import pandas as pd

from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer, RVISATrainer


class Template:
    def __init__(self, args):
        config_dict = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        config = SimpleNamespace(**config_dict)
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config

    def forward(self):
        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'rvisa':
            print("Choosing RVISA multi-task fine-tuning mode.")
            trainer = RVISATrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        else:
            raise 'Should choose a correct reasoning mode: prompt, thor, or rvisa.'

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return

        print("Fine-tuning mode for training.")
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':
    # Load configuration from YAML file
    with open('config/main_config.yaml', 'r', encoding='utf-8') as config_file:
        config_data = yaml.safe_load(config_file)

    args = argparse.Namespace(
        cuda_index=config_data['main']['cuda_index'],
        reasoning=config_data['main']['reasoning'],
        zero_shot=config_data['main']['zero_shot'],
        data_name=config_data['main']['data_name'],
        config=config_data['main']['config'],
        # RVISA-only fields (optional)
        rvisa_data_path=config_data['main'].get('rvisa_data_path', ''),
        rvisa_prompt_style=config_data['main'].get('rvisa_prompt_style', 'th-re'),
        rvisa_alpha=config_data['main'].get('rvisa_alpha', 0.3),
        rvisa_gamma=config_data['main'].get('rvisa_gamma', 0.3),
        rvisa_use_verification=config_data['main'].get('rvisa_use_verification', True),
        rvisa_use_explanation=config_data['main'].get('rvisa_use_explanation', True),
    )

    template = Template(args)
    template.forward()
