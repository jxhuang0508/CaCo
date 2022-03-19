
conda activate caco_pro_ft

# Stage 0: resume network by warm up
#python train.py --name gta2citylabv2_warmup --stage warm_up --freeze_bn --gan LS --lr 2.5e-4 --adv 0.01 --no_resume --used_save_pseudo --path_LP ./Pseudo/train_all


# Stage 1:
#python generate_pseudo_label.py --name gta2citylabv2_warmup_soft --soft --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python calc_prototype.py --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl
#python train.py --name gta2citylabv2_stage1Denoise --used_save_pseudo --ema --proto_rectify --moving_prototype --path_soft Pseudo/gta2citylabv2_warmup_soft --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --proto_consistW 10 --rce --regular_w 0.1


# Stage 2:
#python generate_pseudo_label.py --name gta2citylabv2_stage1Denoise --flip --resume_path ./logs/gta2citylabv2_stage1Denoise/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python train.py --name gta2citylabv2_stage2 --stage stage2 --used_save_pseudo --path_LP Pseudo/gta2citylabv2_stage1Denoise --resume_path ./logs/gta2citylabv2_stage1Denoise/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --S_pseudo 1 --threshold 0.95 --distillation 1 --finetune --lr 6e-4 --student_init simclr --bn_clr --no_resume


# Stage 3:
#python generate_pseudo_label.py --name gta2citylabv2_stage2 --flip --resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast --bn_clr --student_init simclr
python train.py --name gta2citylabv2_stage3 --stage stage3 --used_save_pseudo --path_LP Pseudo/gta2citylabv2_stage2 --resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --S_pseudo 1 --threshold 0.95 --distillation 1 --finetune --lr 6e-4 --student_init simclr --bn_clr --ema_bn

