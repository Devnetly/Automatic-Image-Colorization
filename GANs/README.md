# Usage

## Pre-training the Generator:
Run the `pretrain_gen.py` script to pre-train the generator with supervised learning before adversarial training.
```
python pretrain_gen.py --batch_size 16 --learning_rate 0.001 --epochs 2 --size 256

```
## Adversarial Training

Run training.py to train both the generator and discriminator using cGAN.
```
python training.py --batch_size 16 --learning_rate 0.001 --epochs 2 --lambd 100 --resnet true --size 256
```

If you want to use the pre-trained generator for better results by setting `--resnet=true`
