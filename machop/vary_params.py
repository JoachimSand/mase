import subprocess


for batch_size in [64, 256]:
    for epochs in [5, 10, 20, 30]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            print(f'\n\n\n\n Run with batchsize {batch_size}, epochs {epochs}, learning_rate {learning_rate}')
            subprocess.run(["./ch", "train", "jsc-tiny", "jsc", "--log-level", "warning", "--max-epochs", str(epochs), "--batch-size", str(batch_size), "--accelerator", "gpu"])


for batch_size in [64, 256]:
    for epochs in [5, 10, 20, 30]:
        for learning_rate in [0.001, 0.0001, 0.00001]:
            print(f'\n\n\n\n Run with batchsize {batch_size}, epochs {epochs}, learning_rate {learning_rate}')
            subprocess.run(["./ch", "train", "jsc-ownnetwork", "jsc", "--log-level", "warning", "--max-epochs", str(epochs), "--batch-size", str(batch_size), "--accelerator", "gpu"])


# ./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256
