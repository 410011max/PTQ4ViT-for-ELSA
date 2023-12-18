
## Deit-small
CUDA_VISIBLE_DEVICES=2 python example/test_vit.py --config configs/deit_small.yaml --weight weights/deit_small_patch16_224-cd65a155.pth
CUDA_VISIBLE_DEVICES=3 python example/test_vit.py --config configs/deit_small.yaml --weight elsa_subnet_weights/ELSA_deit_s_u24_2.5G.pth
CUDA_VISIBLE_DEVICES=3 python example/test_vit.py --config configs/deit_small.yaml --weight elsa_subnet_weights/ELSA_deit_s_2G.pth

# Deit-base
CUDA_VISIBLE_DEVICES=2 python example/test_vit.py --config configs/deit_base.yaml --weight weights/deit_base_patch16_224-b5f2ef4d.pth
CUDA_VISIBLE_DEVICES=3 python example/test_vit.py --config configs/deit_base.yaml --weight elsa_subnet_weights/ELSA_deit_b_u24_9.2G.pth
CUDA_VISIBLE_DEVICES=2 python example/test_vit.py --config configs/deit_base.yaml --weight elsa_subnet_weights/ELSA_deit_b_7G.pth
CUDA_VISIBLE_DEVICES=3 python example/test_vit.py --config configs/deit_base.yaml --weight elsa_subnet_weights/ELSA_deit_b_6G.pth

# Swin-small
CUDA_VISIBLE_DEVICES=2 python example/test_vit.py --config configs/swin_small.yaml --weight weights/swin_small_patch4_window7_224.pth
CUDA_VISIBLE_DEVICES=0 python example/test_vit.py --config configs/swin_small.yaml --weight elsa_subnet_weights/ELSA_swin_s_u24_4.6G.pth
CUDA_VISIBLE_DEVICES=0 python example/test_vit.py --config configs/swin_small.yaml --weight elsa_subnet_weights/ELSA_swin_s_4G.pth
CUDA_VISIBLE_DEVICES=0 python example/test_vit.py --config configs/swin_small.yaml --weight elsa_subnet_weights/ELSA_swin_s_3.5G.pth

## Swin-base
CUDA_VISIBLE_DEVICES=1 python example/test_vit.py --config configs/swin_base.yaml --weight weights/swin_base_patch4_window7_224.pth
CUDA_VISIBLE_DEVICES=0 python example/test_vit.py --config configs/swin_base.yaml --weight elsa_subnet_weights/ELSA_swin_b_u24_8G.pth
CUDA_VISIBLE_DEVICES=1 python example/test_vit.py --config configs/swin_base.yaml --weight elsa_subnet_weights/ELSA_swin_b_6G.pth
CUDA_VISIBLE_DEVICES=1 python example/test_vit.py --config configs/swin_base.yaml --weight elsa_subnet_weights/ELSA_swin_b_5.3G.pth





