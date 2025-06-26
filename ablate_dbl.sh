for pos in {0..15}
do
    python eval_agent_with_hooks.py --log-dir pretrained/cartpole_pi_ablate_lstm_rc --model-filename model.npz --n-episodes 20 --seed 1 --prefix ablation_row_$pos --pos $pos --dim 0
done

for pos in {0..4}
do
    python eval_agent_with_hooks.py --log-dir pretrained/cartpole_pi_ablate_lstm_rc --model-filename model.npz --n-episodes 20 --seed 1 --prefix ablation_col_$pos --pos $pos --dim 1
done