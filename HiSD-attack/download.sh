FILE=$1

if [ $FILE == "checkpoint" ]; then

    URL=https://drive.google.com/file/d/1KDrNWLejpo02fcalUOrAJOl1hGoccBKl/
    FILE=./checkpoint_256_celeba-hq.pt
    wget -N $URL -O $FILE

else
    echo "checkpoint."
    exit 1
fi