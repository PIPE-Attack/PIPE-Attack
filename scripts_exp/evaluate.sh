# example shown for lfw, facenet, MRP, diff seed per user

EVAL_FLAGS="--metric euclidean_l2 --model Facnet512 --detector_backend mtcnn" # for facenet as a feature extractor
# EVAL_FLAGS="--metric cosine --model ArcFace --detector_backend retinaface" # for arcface as a feature extractor


python3 evaluation.py --output_dir results/mrp/lfw_facenet_128_d_True --original_dir datasets/mrp/lfw_facenet_128_d_True $EVAL_FLAGS --eval_dset train &
python3 evaluation.py --output_dir results/mrp/lfw_facenet_128_d_True --original_dir datasets/mrp/lfw_facenet_128_d_True $EVAL_FLAGS --eval_dset test 

echo "testing done"