1. Data generation:
   car.sh (simulartions with a leading car, and it will call data_gen.py file, which further calls datagen.scenic)
   car_no.sh (simulations without a leading car, and it will call data_gen_nocar.py, which further calls datagen_nocar.scenic)

2. Training data folder structure:
   Please check the data foler on Alvis-1298, which contains 3000 folders/runs.

2. Controllers training:
   traincnn_turn.sh (train models which contains an extra embedding for turning info)
   # It train_turn.py file can utilizes model architetucre from distance_cte_cnns_turn.py
   # The saved model weights are under folder checkpoints/, please check Alvis-1298

 4. Testing controller:
    test_cnn_controller.scenic
    # the maneuver info has not been passed correctly yet
