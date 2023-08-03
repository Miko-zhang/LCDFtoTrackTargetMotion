import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                    dest="model", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')

# parameters for training
parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default=r'.\train_surf')
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-5)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=156600)
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=0.01)  # recommend 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=1000)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')

args = parser.parse_args()