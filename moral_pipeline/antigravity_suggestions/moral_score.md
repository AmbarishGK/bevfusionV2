# ON LOCAL 4090
cp moral_score.py ~/workspace/amb_ws/bevfusionV2/moral_pipeline/
cd ~/workspace/amb_ws/bevfusionV2/moral_pipeline/

# Run everything
python3 moral_score.py

# Single experiment for quick sanity check first
python3 moral_score.py --experiment "img__B"

# After llm_judge.py produces saves/judge_results/
python3 moral_score.py --merge-rqs

# Regenerate tables without re-scoring (fast)
python3 moral_score.py --tables-only
