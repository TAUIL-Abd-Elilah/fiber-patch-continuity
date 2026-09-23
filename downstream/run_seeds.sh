set -e
cd "D:/Competition/Vesuvius progress prizes/_research_september_20260923/downstream"
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
for seed in 2 3; do
  for arm in A D; do
    $PY -u run_fit.py --arm $arm --steps 3000 --seed $((20260923 + seed)) --name s${seed}_${arm}_3000 > runs_s${seed}_${arm}.log 2>&1
    $PY evaluate_heldout.py --run s${seed}_${arm}_3000 >> runs_s${seed}_${arm}.log 2>&1
  done
done
echo ALL_DONE
