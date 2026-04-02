I tested whether OT-based couplings produce more efficient learned generative flows, and I quantified when minibatch OT helps or hurts.

python generate_data.py --n_samples 20000 --out_dir data --stem toy_moons
python train.py --data_path data/toy_moons.pt --save_dir runs/toy_moons_ot --coupling ot
python train.py --data_path data/toy_moons.pt --save_dir runs/toy_moons_ind --coupling independent