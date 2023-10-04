# runner.py
import os
learning_rate_list = [0.00001,0.000012, 0.000013, 0.000015]
for learning_rate in learning_rate_list:
    # round learning rate to 6 decimal places
    learning_rate = round(learning_rate, 7)
    attention_probs_dropout_prob = 0.007 
    test_mode = True
    model_name = 'debertav3large'
    command = f"python tuned-debertav3-lgbm-autocorrect_Merge_model.py --learning_rate {learning_rate} --attention_probs_dropout_prob {attention_probs_dropout_prob} --test_mode {test_mode}| tee log_file/hps_fus_lr_{learning_rate}_att_{attention_probs_dropout_prob}.log"
    os.system(command)