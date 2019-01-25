FILE=$1

if [[ $FILE != "ThermalWorld" ]]; then
    echo "Available dataset is: ThermalWorld"
    exit 1
fi

URL=http://zefirus.org/datasets/models/ThermalGAN/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
