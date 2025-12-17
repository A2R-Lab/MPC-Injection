Commands to generate these plots:

python reward_surfaces/scripts/generate_plane_jobs.py logs/SAC-MPC-walker-stand_only_reward/1st_run/walker-walk-SAC-MPC-20251208-145319-percentage-0pct/best_model/ plots/sac_mpc_walker_stand_only_reward_0_reward_surface/ --grid-size=31 --magnitude=0.5 --num-episodes=25
python reward_surfaces/scripts/run_jobs_multiproc.py --num-cpus=8 plots/sac_mpc_walker_stand_only_reward_0_reward_surface/jobs.sh 
python reward_surfaces/scripts/job_results_to_csv.py plots/sac_mpc_walker_stand_only_reward_0_reward_surface/
python reward_surfaces/scripts/plot_plane.py plots/sac_mpc_walker_stand_only_reward_0_reward_surface/results.csv --outname=plots/sac_mpc_walker_stand_only_reward_0_reward_surface/surface --env-name="SAC-MPC Walker Walk 0pct" --type=all
