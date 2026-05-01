# Tricks & constraints

Working agreements between user and agent. Concise — append; do not bloat.

1. **Always ask which tmux session to use if the user hasn't specified.**
   GPU allocations expire silently; the wrong session is dead, and a fresh
   `srun` may live in a session number the agent can't guess. Ask once at
   the start of any session that involves running on GPU.

2. **Exploration: sweep faster, record faithfully, don't optimise mid-stage.**
   Each tier / config is a hypothesis. Land the result honestly (single
   seed is OK; faithfulness > polish), move to the next variant, *then*
   look back at the whole sweep and decide where to push. Don't burn
   cycles tuning one underperforming intermediate point.

3. **Align with the previous performance / Stage 1 difflogic recipe.**
   When a new model class is introduced (streaming, etc.), keep the parts
   that aren't the new contribution close to the MNIST difflogic baseline:
   `hidden_dim ≈ 8000`, `num_layers ≈ 6`, `tau` scaled to keep
   `tau / group_size ≈ 0.0125`, Adam `lr=0.01`. Target params ≈ 800K +
   modest headroom for temporal-context capacity. This minimises
   architectural divergence so any performance delta is attributable to
   the new contribution, not to incidental hyperparameter drift.
