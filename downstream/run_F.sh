set -e
cd "D:/Competition/Vesuvius progress prizes/_research_september_20260923/downstream"
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
for s in 1:20260923 2:20260925 3:20260926; do
  n=${s%%:*}; seed=${s##*:}
  $PY -u run_fit.py --arm F --steps 3000 --seed $seed --name s${n}_F_3000 > runs_s${n}_F.log 2>&1
  $PY evaluate_heldout.py --run s${n}_F_3000 >> runs_s${n}_F.log 2>&1
done
echo ALL_DONE
