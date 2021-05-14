# CS726-Project
## Usage
Cloning the repository: git clone https://github.com/VarshithaReddy8/CS726-Project.git

StarGAN:

1. Dataset
Download CelebA dataset  
$ bash download.sh celeba

2. Pretrained Models
Download pretrained starGAN models -  
$ bash download.sh pretrained-celeba-128x128 or  
$ bash download.sh pretrained-celeba-256x256  
Save these models in stargan_celeba_128 and stargan_celeba_256 respectively.

3. Testing attacks 
To test StarGAN on CelebA  
$ python3 main.py --mode test_attack --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
                 --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
                 
HiSD

1. Dataset
Download CelebA dataset and put the images in dataset folder.

2. Model Configuration
Download pretrained checkpoint checkpoint_256_celeba-hq.pt from 
[Google Drive](https://drive.google.com/file/d/1KDrNWLejpo02fcalUOrAJOl1hGoccBKl/view?usp=sharing) this link to the root directory.

3. Testing attacks
$ python3 main.py --mode test_attack --dataset_dir dataset

For more options check main.py files of the models.

