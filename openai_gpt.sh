# Folder and Dataset name
folder="DWUG"
dataset="English"

# Download data from zenodo
wget https://zenodo.org/record/7387261/files/dwug_en.zip?download=1
unzip dwug_en.zip\?download\=1
rm dwug_en.zip\?download\=1

# Create directory for data
mkdir "${folder}"
mv dwug_en "${folder}/${dataset}"

# targets
targets=("afternoon_nn" "donkey_nn" "land_nn" "player_nn" "stroke_vb" "attack_nn" "edge_nn" "lane_nn" "prop_nn" "thump_nn" "bag_nn" "face_nn" "lass_nn" "quilt_nn" "tip_vb" "ball_nn" "fiction_nn" "maxim_nn" "rag_nn" "tree_nn" "bar_nn" "gas_nn" "multitude_nn" "rally_nn" "twist_nn" "bit_nn" "graft_nn" "ounce_nn" "record_nn" "word_nn" "chairman_nn" "grain_nn" "part_nn" "relationship_nn" "chef_nn" "head_nn" "pick_vb" "risk_nn" "circle_vb" "heel_nn" "pin_vb" "savage_nn" "contemplation_nn" "include_vb" "plane_nn" "stab_nn")

for target in "${targets[@]}"
do
    # collect data from chatgpt
    python openai_gpt.py --target "${target}" --dataset "${folder}/${dataset}"

    # Avoid issues due to too many requests in a short period of time
    #sleep 120
done
