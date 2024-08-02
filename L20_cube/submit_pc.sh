nres=1

mkdir -p plots
mkdir -p data

echo "variable xx uloop $nres" > in.variables
echo >> in.variables
printf  "variable vseed universe " >> in.variables
for (( x=0; x<$nres; x++ ))
do
    printf "%i " $RANDOM >> in.variables

done

python3 polymer.py

time mpirun -n 1 lmp_mpi -in in_summit.lammps > out.run

rm log.*