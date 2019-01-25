FILE=$1

echo "Note: available model is ThermalGAN"

echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
URL=http://zefirus.org/datasets/models/ThermalGAN/$FILE.pth

wget -N $URL -O $MODEL_FILE


