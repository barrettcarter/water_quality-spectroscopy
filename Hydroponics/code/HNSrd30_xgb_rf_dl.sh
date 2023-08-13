module load conda

conda activate humhpc

python /blue/ezbean/jbarrett.carter/water_quality-spectroscopy/Hydroponics/code/abs-wq_an_HNSr_xgb.py

python /blue/ezbean/jbarrett.carter/water_quality-spectroscopy/Hydroponics/code/abs-wq_an_HNSr_RF-PCA.py

conda activate tf_gpu

python /blue/ezbean/jbarrett.carter/water_quality-spectroscopy/Hydroponics/code/abs-wq_an_HNSr_DL-lenet5.py