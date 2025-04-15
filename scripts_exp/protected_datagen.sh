# Same s and different s setting to generate the protected dataset

# Possible datasets: lfw_arcface, celeba_arcface, ffhq_arcface and lfw_facenet, celeba_facenet, ffhq_facenet
# Possible settings: s, d           # same or different seed per user # same or different seed per user
# Possible known_r: yes, no         # corresponds to known s setting (no implies baseline run while yes implies PIPE)
# Possible schemes: datagen_lwe_fe.py , datagen_mrp.py
python datagen_lwe_fe.py --dataset lfw_arcface --setting d --known_r yes