# SignalingTraj


# Training:
  Running the following command to train SignalingTraj.
  
  **Chengdu dataset:**     python multi_main.py --dataset Chengdu --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **operator0 dataset:**     python multi_main.py --dataset operator0 --embed_mode vgae --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **operator1 dataset:**     python multi_main.py --dataset operator1 --embed_mode vgae --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **operator3 dataset:**     python multi_main.py --dataset operator3 --embed_mode vgae --diff_T 500 --pre_trained_dim 64 --rdcl 10

# Generate data:
  After training the SignalingTraj, run the following command to generate the road network-constrained trajectory.

  **Chengdu dataset:**     python generate_data.py --dataset Chengdu --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **operator0 dataset:**     python generate_data.py --dataset operator0 --embed_mode vgae --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **operator1 dataset:**     python generate_data.py --dataset operator1 --embed_mode vgae --diff_T 500 --pre_trained_dim 64 --rdcl 10
  
  **operator3 dataset:**     python generate_data.py --dataset operator3 --embed_mode vgae --diff_T 500 --pre_trained_dim 64 --rdcl 10
