TIMESTAMP="$(date +'%m%d_%H%M%S')"
echo "TIMESTAMP: $TIMESTAMP"
./train.sh "$TIMESTAMP" > terminal/$TIMESTAMP.out 2> terminal/$TIMESTAMP.err &