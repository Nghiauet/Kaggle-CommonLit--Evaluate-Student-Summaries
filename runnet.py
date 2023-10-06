# runner.py
import os
learning_rate_list = [0.000014,0.000015, 0.000017, 0.000018,0.00005,0.000021,0.000022]
for learning_rate in learning_rate_list:
    # round learning rate to 6 decimal places
    learning_rate = round(learning_rate, 7)
    attention_probs_dropout_prob = 0.007
    test_mode = False
    model_name = 'debertav3base'
    command = f"python tuned-debertav3-lgbm-autocorrect_Merge_model.py --learning_rate {learning_rate} --attention_probs_dropout_prob {attention_probs_dropout_prob} --test_mode {test_mode} --model_name {model_name}| tee log_file/{model_name}_{learning_rate}_att_{attention_probs_dropout_prob}.log"
    os.system(command)