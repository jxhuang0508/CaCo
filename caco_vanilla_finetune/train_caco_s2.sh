## Generate pseudo labels ---------------------------------
#python generate_plabel_cityscapes_advent_caco.py  --restore-from ./snapshots/caco_stage1/caco_stage1.pth

### Fine-tune network ---------------------------------
#python train_ft_advent_caco.py --snapshot-dir ./snapshots/CaCo_stage2 \
#--restore-from ./snapshots/caco_stage1/caco_stage1.pth \
#--drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.1 --lambda-adv-target1 0 \
#--lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 \
#--max-value 7 --gpu-ids 0,1,2 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False --save-pred-every 1000


### test best
for i in {1000..100000..1000}
do
	echo "TEST $i MODEL"
	CUDA_VISIBLE_DEVICES=0,1 python evaluate_cityscapes_advent_best.py --restore-from ./snapshots/CaCo_stage2/GTA5_$i.pth
done
