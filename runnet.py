# runner.py
import os
learning_rate_list = [
                    #   0.000006,
                      0.000007,
                      0.000008,
                      0.000012,    
                      0.000015,
                      0.000016,
                      ]
for learning_rate in learning_rate_list:
    # round learning rate to 6 decimal places
    learning_rate = round(learning_rate, 7)
    attention_probs_dropout_prob = 0.007
    test_mode = False
    model_name = 'debertav3large'
    model_name_to_save = model_name.replace('/', '_')
    command = f"python tuned-debertav3-Merge_clean_text_1010.py --learning_rate {learning_rate} --attention_probs_dropout_prob {attention_probs_dropout_prob} --test_mode {test_mode} --model_name {model_name}| tee log_file/{model_name_to_save}_{learning_rate}_att_{attention_probs_dropout_prob}_text_clean.log"
    os.system(command)