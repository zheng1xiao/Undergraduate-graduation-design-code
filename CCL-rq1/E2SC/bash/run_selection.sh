cd $WORKDIR

datain="$WORKDIR/resources/datasets"
out="$WORKDIR/resources/outsel2"

mkdir -p $out

datasets=(aisopos_ntua_2L)
#methods=(cnn) # enn icf lssm lsbo ldis cdis xldis drop3 psdsp ib3 cis egdis) 
methods=(e2sc-1 e2sc-2)
#methods=(drop3) 

for d in ${datasets[@]};
do
    echo $d ; 
    for method in ${methods[@]} 
    do
        echo $method ;
        python3 run\_generateSplit.py -d $d -m $method --datain $datain --out $out;
    done;
done;