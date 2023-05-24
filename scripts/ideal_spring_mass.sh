dynamics="ideal_spring_mass"
traj_num=50
dynamics_model_input_dim=2
dynamics_model_output_dim=2
dynamics_model_hidden_dim=100
traj_model_input_dim=2
traj_model_output_dim=1
traj_model_hidden_dim=100
learning_rate=1e-3

for noise in {0.01,}
do
for seed in {2023,}
do
python train_contrastive_conservation.py --dynamics $dynamics --dynamics_samples $traj_num --dynamics_noise $noise \
--traj_model_def MLP  --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--epochs 1000 --batch_size 10 --learning_rate $learning_rate --seed $seed \
--save_name "./saved_models/${dynamics}_traj_repre_$traj_model_output_dim.tar"

python train_dynamics.py --dynamics $dynamics --dynamics_samples $traj_num --dynamics_noise $noise \
--dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode project_with_knowngx --load_traj_model "./saved_models/${dynamics}_traj_repre_$traj_model_output_dim.tar" \
--epochs 1000 --batch_size 100 --learning_rate $learning_rate --seed $seed \
--save_name "./saved_models/${dynamics}_dynamics_traj_consv.tar"

python evaluate_dynamics.py --dynamics $dynamics --dynamics_noise $noise \
--dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode project_with_knowngx --load_traj_model "./saved_models/${dynamics}_traj_repre_$traj_model_output_dim.tar" \
--load_dynamics_model "./saved_models/${dynamics}_dynamics_traj_consv.tar" \
--eval_t_end 50.0 --seed $seed 

python train_dynamics.py --dynamics $dynamics --dynamics_samples $traj_num --dynamics_noise $noise \
--dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode default --load_traj_model "" \
--epochs 1000 --batch_size 100 --learning_rate $learning_rate --seed $seed \
--save_name "./saved_models/${dynamics}_dynamics_noconsv.tar"

python evaluate_dynamics.py --dynamics $dynamics --dynamics_noise $noise \
--dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode default --load_traj_model "" \
--load_dynamics_model "./saved_models/${dynamics}_dynamics_noconsv.tar" \
--eval_t_end 50.0 --seed $seed 
done
done