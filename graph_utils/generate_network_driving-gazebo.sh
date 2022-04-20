echo "################"
echo "gaia"
python generate_networks.py gaia --experiment driving_gazebo --upload_capacity 1e10 --download_capacity 1e10
echo "################"
echo "amazon_us"
python generate_networks.py amazon_us --experiment driving_gazebo --upload_capacity 1e10 --download_capacity 1e10