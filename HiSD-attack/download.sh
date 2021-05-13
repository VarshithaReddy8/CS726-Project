FILE=$1

if [ $FILE == "checkpoint" ]; then

    URL=https://drive.google.com/file/d/1KDrNWLejpo02fcalUOrAJOl1hGoccBKl/
    FILE=./checkpoint_256_celeba-hq.pt
    # mkdir -p ./data/
    wget -N $URL -O $FILE
    # unzip $ZIP_FILE -d ./data/
    # rm $ZIP_FILE

else
    echo "checkpoint."
    exit 1
fi