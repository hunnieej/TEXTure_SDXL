import pyrallis

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure


@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = TEXTure(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
        #10번 painting 다 하고 나서 obj, v, vt, f 저장(model export mesh 함수 작동)
    else:
        trainer.paint()


if __name__ == '__main__':
    main()
