FILE=$1

if [[ $FILE != "ThermalGAN" ]]; then
    echo "Available dataset is: ThermalGAN"
    exit 1
fi

URL=http://zefirus.org/datasets/models/ThermalGAN/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
