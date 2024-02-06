import subprocess


for batch_size in [32, 64, 256, 1028]:
    print(f'\n\n\n\n Run with batchsize {batch_size}')
    subprocess.run(["./ch", "train", "jsc-tiny", "jsc", "--max-epochs", "10", "--batch-size", str(batch_size)])




# ./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256
