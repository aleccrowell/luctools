for f in ./Jill_analyses/*.csv;
do
python3 analyse_luctraces.py -i "$f" -s 1
done

