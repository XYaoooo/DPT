#### 1. How To Install
You can check the `requirements.txt` for the required packages.
****

#### 2. Reproduce Results
Task name can be changed

```python
CUDA_VISIBLE_DEVICES=1 python main_ddp.py --datasets='rte'  --model_name=t5-base --enc_prompt_tokens 100 -ts 16 -e 100 --bottle_neck 10
```

## Reference
If you find our work helpful, please consider citing our paper:
```bibtex
@inproceedings{dpt2023emnlp,
    title = "Decomposed Prompt Tuning via Low-Rank Reparameterization",
    author = "Xiao, Yao and Xu, Lu and Li, Jiaxi and Lu, Wei and Li, Xiaoli",
    booktitle = "Proceedings of EMNLP Finding ",
    year = "2023",
}
```
Our code is base on MPT, you can cite it by:
```bibtex
@inproceedings{
wang2023multitask,
title={Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning},
author={Zhen Wang and Rameswar Panda and Leonid Karlinsky and Rogerio Feris and Huan Sun and Yoon Kim},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
}
```
