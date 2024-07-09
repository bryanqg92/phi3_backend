while read requirement; do
    pip install $requirement
done < requirements.txt