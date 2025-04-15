EVAL_FLAGS="--metric euclidean_l2 --model Facenet512 --detector_backend mtcnn --output_dir results --original_dir DataGen/databases/celeba_pp --samples 1000"
python3 evaluation.py $EVAL_FLAGS  --prot_scheme unprotected --same_r none --eval_dset train &
# python3 evaluation.py $EVAL_FLAGS  --prot_scheme unprotected --same_r none --eval_dset test 

# python3 evaluation.py $EVAL_FLAGS  --prot_scheme mrp --same_r true --eval_dset train &
# python3 evaluation.py $EVAL_FLAGS  --prot_scheme mrp --same_r true  --eval_dset test 

# python3 evaluation.py $EVAL_FLAGS  --prot_scheme mrp --same_r false --eval_dset train &
# python3 evaluation.py $EVAL_FLAGS  --prot_scheme mrp --same_r false --eval_dset test 

# python3 evaluation.py $EVAL_FLAGS  --prot_scheme fe --same_r true --eval_dset train &
# python3 evaluation.py $EVAL_FLAGS  --prot_scheme fe --same_r true --eval_dset test 

# python3 evaluation.py $EVAL_FLAGS  --prot_scheme fe --same_r false --eval_dset train &
# python3 evaluation.py $EVAL_FLAGS  --prot_scheme fe --same_r false --eval_dset test 
echo "testing done"