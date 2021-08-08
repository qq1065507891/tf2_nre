import argparse


from src.utils.config import config
from src.training import train
from src.predict import predict

parser = argparse.ArgumentParser(description='关系抽取')
parser.add_argument('--model_name', type=str, default='TextCNN', help='model name')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model_name if args.model_name else config.model_name
    train(config, model_name)


