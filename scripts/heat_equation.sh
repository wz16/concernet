dynamics="heat_equation"
traj_num=200
dynamics_model_input_dim=9
dynamics_model_output_dim=9
dynamics_model_hidden_dim=100
autoencoder_model_input_dim=101
autoencoder_model_latent_dim=9
traj_model_input_dim=9
traj_model_output_dim=1
traj_model_hidden_dim=100
learning_rate=1e-3
for noise in {0.01,}
do
for seed in {2023,}
do

# python train_autoencoder.py --dynamics $dynamics --dynamics_samples $traj_num --dynamics_noise $noise \
# --autoencoder_model_def AE1D --autoencoder_model_input_dim $autoencoder_model_input_dim --autoencoder_model_latent_dim $autoencoder_model_latent_dim \
# --epochs 200 --batch_size 100 --learning_rate $learning_rate --save_name "./saved_models/${dynamics}_autoencoder.tar" --seed $seed

python train_contrastive_conservation.py --dynamics $dynamics --dynamics_samples $traj_num --dynamics_noise $noise \
--ae_mode --autoencoder_model_def AE1D --autoencoder_model_input_dim $autoencoder_model_input_dim --autoencoder_model_latent_dim $autoencoder_model_latent_dim \
--load_ae_model "./saved_models/${dynamics}_autoencoder.tar" \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--seed=$seed --epochs 1000 --batch_size 10 --learning_rate $learning_rate --save_name "./saved_models/${dynamics}_traj_repre_$traj_model_output_dim.tar"

python train_dynamics.py --dynamics $dynamics --dynamics_noise $noise --dynamics_samples $traj_num --dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--ae_mode --autoencoder_model_def AE1D --autoencoder_model_input_dim $autoencoder_model_input_dim --autoencoder_model_latent_dim $autoencoder_model_latent_dim \
--load_ae_model "./saved_models/${dynamics}_autoencoder.tar" \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode project_with_knowngx --load_traj_model "./saved_models/${dynamics}_traj_repre_$traj_model_output_dim.tar" \
--seed=$seed --epochs 200 --batch_size 20 --learning_rate $learning_rate --save_name "./saved_models/${dynamics}_dynamics_traj_consv.tar"

python evaluate_dynamics.py --dynamics $dynamics --dynamics_noise $noise --dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--load_dynamics_model "./saved_models/${dynamics}_dynamics_traj_consv.tar" \
--ae_mode --autoencoder_model_def AE1D --autoencoder_model_input_dim $autoencoder_model_input_dim --autoencoder_model_latent_dim $autoencoder_model_latent_dim \
--load_ae_model "./saved_models/${dynamics}_autoencoder.tar" \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode project_with_knowngx --load_traj_model "./saved_models/${dynamics}_traj_repre_$traj_model_output_dim.tar" \
--seed=$seed --eval_t_end 1.0 
done

python train_dynamics.py --dynamics $dynamics --dynamics_noise $noise --dynamics_samples $traj_num --dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--ae_mode --autoencoder_model_def AE1D --autoencoder_model_input_dim $autoencoder_model_input_dim --autoencoder_model_latent_dim $autoencoder_model_latent_dim \
--load_ae_model "./saved_models/${dynamics}_autoencoder.tar" \
--traj_model_def MLP --traj_model_input_dim $traj_model_input_dim --traj_model_output_dim $traj_model_output_dim --traj_model_hidden_dim $traj_model_hidden_dim \
--wrapper_mode default --load_traj_model "" \
--seed=$seed --epochs 200 --batch_size 20 --learning_rate $learning_rate --save_name "./saved_models/${dynamics}_dynamics_noconsv.tar"

python evaluate_dynamics.py --dynamics $dynamics --dynamics_noise $noise --dynamics_model_def MLP --dynamics_model_input_dim $dynamics_model_input_dim --dynamics_model_output_dim $dynamics_model_output_dim --dynamics_model_hidden_dim $dynamics_model_hidden_dim \
--load_dynamics_model "./saved_models/${dynamics}_dynamics_noconsv.tar" \
--ae_mode --autoencoder_model_def AE1D --autoencoder_model_input_dim $autoencoder_model_input_dim --autoencoder_model_latent_dim $autoencoder_model_latent_dim \
--load_ae_model "./saved_models/${dynamics}_autoencoder.tar" \
--wrapper_mode default \
--seed=$seed --eval_t_end 1.0

done