set -e
cd "D:/Competition/Vesuvius progress prizes/_research_september_20260923/downstream"
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
for arm in A D P; do
  $PY -u run_fit.py --arm $arm --steps 3000 --name s1_${arm}_3000 > runs_s1_${arm}.log 2>&1
  $PY evaluate_heldout.py --run s1_${arm}_3000 >> runs_s1_${arm}.log 2>&1
done
echo ALL_DONE
