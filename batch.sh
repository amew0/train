TIMESTAMP="$(date +'%m%d_%H%M%S')"
echo "TIMESTAMP: $TIMESTAMP"
./train_ds.sh "$TIMESTAMP" > terminal/$TIMESTAMP.out 2> terminal/$TIMESTAMP.err & 
# ./train_acc.sh "$TIMESTAMP"